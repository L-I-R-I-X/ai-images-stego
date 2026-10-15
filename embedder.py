#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embedder.py — утилита встраивания цифрового водяного знака (ЦВЗ).

Схема: DWT (HL1) -> DCT 8x8 -> QIM в коэффициенте [4,4].
Полезная нагрузка: ядро (magic + 4-байтовый ID) + подпись Ed25519 = 70 байт,
защита Reed-Solomon RS(100,70) -> 100 байт -> 800 бит.

Встраивание происходит ТОЛЬКО в канал яркости (Y). Результат сохраняется
исключительно в формате без потерь (PNG/TIFF/BMP).

Константы протокола должны совпадать с detector.py.
"""

import argparse
import hashlib
import os
import struct
import sys

import cv2
import numpy as np
import pywt
import reedsolo
from scipy.fft import dctn, idctn

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# --------------------------------------------------------------------------
# КОНСТАНТЫ ПРОТОКОЛА (обязаны совпадать с detector.py)
# --------------------------------------------------------------------------
MAGIC = b"\x49\x41"          # 0x49 0x41 ("IA") — маркер начала
CORE_LEN = 6                 # 2 (magic) + 4 (ID)
SIG_LEN = 64                 # Ed25519 signature
PAYLOAD_LEN = CORE_LEN + SIG_LEN          # 70
ECC_LEN = 30                 # RS(100,70): 30 байт избыточности
CODE_LEN = PAYLOAD_LEN + ECC_LEN          # 100
BITS = CODE_LEN * 8          # 800
WAVELET = "haar"             # тип вейвлета
DETAIL_INDEX = 0             # 0 = cH (HL1)
BLOCK = 8                    # размер DCT-блока
COEF_POS = (4, 4)            # закреплённая координата коэффициента
DEFAULT_DELTA = 30           # шаг квантования QIM
ALLOWED_EXT = {".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------
# Работа с изображениями (устойчиво к путям в UTF-8, напр. кириллице)
# --------------------------------------------------------------------------
def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError("не удалось прочитать файл: %s" % path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось декодировать изображение: %s" % path)
    return img


def save_image(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("не удалось закодировать изображение в формат %s" % ext)
    buf.tofile(path)


# --------------------------------------------------------------------------
# Payload: сообщение -> 800 бит
# --------------------------------------------------------------------------
def message_to_id4(message):
    """Строка/число -> 4 байта. Число -> big-endian; иначе sha256[:4]."""
    s = str(message)
    try:
        n = int(s, 10)
    except ValueError:
        n = None
    if n is not None and 0 <= n <= 0xFFFFFFFF:
        return struct.pack(">I", n)
    return hashlib.sha256(s.encode("utf-8")).digest()[:4]


def build_bits(message, private_key):
    id4 = message_to_id4(message)
    core = MAGIC + id4
    signature = private_key.sign(core)
    payload = core + signature
    if len(payload) != PAYLOAD_LEN:
        raise RuntimeError("неверная длина payload: %d" % len(payload))

    rs = reedsolo.RSCodec(ECC_LEN)
    code = bytes(rs.encode(payload))
    if len(code) != CODE_LEN:
        raise RuntimeError("неверная длина RS-кода: %d" % len(code))

    bits = np.unpackbits(np.frombuffer(code, dtype=np.uint8))
    return bits, core


# --------------------------------------------------------------------------
# QIM
# --------------------------------------------------------------------------
def qim_embed(c, bit, delta):
    """Бит 0 -> ближайшее чётное кратное delta; бит 1 -> ближайшее нечётное."""
    if bit == 0:
        step = 2.0 * delta
        return round(c / step) * step
    k = round((c / delta - 1.0) / 2.0)
    return (2 * k + 1) * delta


# --------------------------------------------------------------------------
# Основной алгоритм встраивания
# --------------------------------------------------------------------------
def embed(img, bits, delta):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float64)

    coeffs = pywt.dwt2(y, WAVELET, mode="periodization")
    cA, (cH, cV, cD) = coeffs
    detail = [cH, cV, cD][DETAIL_INDEX]

    hb, wb = detail.shape
    blocks_per_col = hb // BLOCK
    blocks_per_row = wb // BLOCK
    capacity = blocks_per_col * blocks_per_row
    if capacity < BITS:
        raise ValueError(
            "изображение слишком мало: поддиапазон HL1 %dx%d вмещает %d блоков, "
            "требуется %d. Увеличьте разрешение изображения."
            % (wb, hb, capacity, BITS)
        )

    ri, ci = COEF_POS
    idx = 0
    for r in range(blocks_per_col):
        if idx >= BITS:
            break
        for c in range(blocks_per_row):
            if idx >= BITS:
                break
            block = detail[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK]
            # norm="ortho" — ортонормальный DCT: правка коэффициента напрямую
            # влияет на пиксели блока и переживает округление до uint8.
            bd = dctn(block, type=2, norm="ortho")
            bd[ri, ci] = qim_embed(bd[ri, ci], int(bits[idx]), delta)
            detail[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK] = idctn(
                bd, type=2, norm="ortho"
            )
            idx += 1

    y_new = pywt.idwt2(coeffs, WAVELET, mode="periodization")
    y_new = np.clip(np.round(y_new), 0, 255).astype(np.uint8)

    yuv[:, :, 0] = y_new
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# --------------------------------------------------------------------------
# Ключи
# --------------------------------------------------------------------------
def load_or_generate_key(path):
    if path:
        with open(path, "rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("ключ %s не является закрытым ключом Ed25519" % path)
        return key, None

    key = Ed25519PrivateKey.generate()
    priv_path = os.path.join(os.getcwd(), "private_key.pem")
    pub_path = os.path.join(os.getcwd(), "public_key.pem")
    with open(priv_path, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    with open(pub_path, "wb") as f:
        f.write(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
    return key, (priv_path, pub_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Встраивание ЦВЗ в изображение (DWT-DCT-QIM + Ed25519 + RS)."
    )
    p.add_argument("-i", "--input", required=True, help="исходное изображение")
    p.add_argument("-m", "--message", required=True, help="ID автора (число или строка)")
    p.add_argument("-o", "--output", required=True, help="маркированное изображение (PNG/TIFF/BMP)")
    p.add_argument("-k", "--key", help="путь к закрытому ключу Ed25519 (PEM)")
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="шаг квантования QIM (по умолчанию 30)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    ext = os.path.splitext(args.output)[1].lower()
    if ext not in ALLOWED_EXT:
        print("[FAIL] Выходной формат '%s' недопустим. Используйте PNG/TIFF/BMP "
              "(сохранение без потерь)." % (ext or "<нет>"))
        return 2

    if args.delta <= 0:
        print("[FAIL] Параметр --delta должен быть положительным.")
        return 2

    try:
        private_key, generated = load_or_generate_key(args.key)
    except Exception as exc:
        print("[FAIL] Ошибка ключа: %s" % exc)
        return 2

    if generated:
        print("[i] Сгенерирована новая пара ключей:")
        print("    закрытый ключ: %s" % generated[0])
        print("    открытый ключ: %s" % generated[1])

    try:
        img = load_image(args.input)
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 2

    try:
        bits, core = build_bits(args.message, private_key)
    except Exception as exc:
        print("[FAIL] Ошибка формирования payload: %s" % exc)
        return 2

    try:
        marked = embed(img, bits, args.delta)
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 2

    try:
        save_image(args.output, marked)
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 2

    id_value = struct.unpack(">I", core[2:6])[0]
    print("[OK] Метка встроена.")
    print("    ID: %d" % id_value)
    print("    Канал: Y, поддиапазон HL1, DCT %dx%d, коэф. %s, delta=%g"
          % (BLOCK, BLOCK, COEF_POS, args.delta))
    print("    Файл: %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
