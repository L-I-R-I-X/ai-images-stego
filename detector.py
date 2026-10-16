#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detector.py — утилита слепого извлечения и верификации ЦВЗ.

Обратный конвейер к embedder.py: YUV -> Y -> DWT (HL1) -> DCT 8x8 ->
считывание QIM-коэффициента [4,4] -> 800 бит -> 100 байт -> RS(100,70) ->
проверка подписи Ed25519 -> вывод ID.

Константы протокола должны совпадать с embedder.py.
"""

import argparse
import os
import struct
import sys

import cv2
import numpy as np
import pywt
import reedsolo
from scipy.fft import dctn

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# --------------------------------------------------------------------------
# КОНСТАНТЫ ПРОТОКОЛА (обязаны совпадать с embedder.py)
# --------------------------------------------------------------------------
MAGIC = b"\x49\x41"          # 0x49 0x41 ("IA")
CORE_LEN = 6
SIG_LEN = 64
PAYLOAD_LEN = CORE_LEN + SIG_LEN          # 70
ECC_LEN = 30
CODE_LEN = PAYLOAD_LEN + ECC_LEN          # 100
BITS = CODE_LEN * 8          # 800
WAVELET = "haar"
DETAIL_INDEX = 0             # 0 = cH (HL1)
BLOCK = 8
COEF_POS = (4, 4)
DEFAULT_DELTA = 30
ALLOWED_EXT = {".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------
# Работа с изображениями (устойчиво к путям в UTF-8)
# --------------------------------------------------------------------------
def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError("не удалось прочитать файл: %s" % path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось декодировать изображение: %s" % path)
    return img


# --------------------------------------------------------------------------
# QIM детекция
# --------------------------------------------------------------------------
def qim_detect(c, delta):
    """Бит 0, если коэффициент ближе к чётной сетке кратных delta, иначе 1."""
    step = 2.0 * delta
    even = round(c / step) * step
    k = round((c / delta - 1.0) / 2.0)
    odd = (2 * k + 1) * delta
    return 0 if abs(c - even) <= abs(c - odd) else 1


# --------------------------------------------------------------------------
# Извлечение 800 бит
# --------------------------------------------------------------------------
def extract_bits(img, delta):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float64)

    coeffs = pywt.dwt2(y, WAVELET, mode="periodization")
    detail = coeffs[1][DETAIL_INDEX]

    hb, wb = detail.shape
    blocks_per_col = hb // BLOCK
    blocks_per_row = wb // BLOCK
    capacity = blocks_per_col * blocks_per_row
    if capacity < BITS:
        raise ValueError(
            "изображение слишком мало: поддиапазон HL1 %dx%d вмещает %d блоков, "
            "требуется %d." % (wb, hb, capacity, BITS)
        )

    ri, ci = COEF_POS
    bits = np.empty(BITS, dtype=np.uint8)
    idx = 0
    for r in range(blocks_per_col):
        if idx >= BITS:
            break
        for c in range(blocks_per_row):
            if idx >= BITS:
                break
            block = detail[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK]
            bd = dctn(block, type=2, norm="ortho")
            bits[idx] = qim_detect(bd[ri, ci], delta)
            idx += 1
    return bits


# --------------------------------------------------------------------------
# Reed-Solomon + криптопроверка
# --------------------------------------------------------------------------
def rs_decode(code):
    rs = reedsolo.RSCodec(ECC_LEN)
    result = rs.decode(code)
    if isinstance(result, tuple):
        result = result[0]
    return bytes(result)


def load_public_key(path):
    with open(path, "rb") as f:
        key = serialization.load_pem_public_key(f.read())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("ключ %s не является открытым ключом Ed25519" % path)
    return key


# --------------------------------------------------------------------------
# Извлечение и верификация (одна попытка на заданной ориентации)
# --------------------------------------------------------------------------
STATUS_RS = "rs"          # Reed-Solomon не восстановил пакет
STATUS_MAGIC = "magic"    # пакет декодирован, но маркер не совпал
STATUS_SIG = "sig"        # маркер на месте, подпись недействительна
STATUS_OK = "ok"          # подпись подтверждена
_STATUS_RANK = {STATUS_RS: 0, STATUS_MAGIC: 1, STATUS_SIG: 2, STATUS_OK: 3}

# Ориентации для перебора: (градусы, флаг cv2.rotate или None для исходной)
ROTATIONS = (
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
)


def attempt_withdraw(img, delta, public_key):
    """Одна попытка извлечь и проверить метку из изображения.

    Возвращает (status, id_value | None). Исключение ValueError (например,
    изображение слишком мало) пробрасывается наружу — поворот его не исправит.
    """
    bits = extract_bits(img, delta)          # может бросить ValueError
    code = np.packbits(bits).tobytes()

    try:
        payload = rs_decode(code)
    except reedsolo.ReedSolomonError:
        return STATUS_RS, None

    if len(payload) != PAYLOAD_LEN:
        return STATUS_RS, None

    core = payload[:CORE_LEN]
    signature = payload[CORE_LEN:CORE_LEN + SIG_LEN]

    if core[:2] != MAGIC:
        return STATUS_MAGIC, None

    try:
        public_key.verify(signature, core)
    except InvalidSignature:
        return STATUS_SIG, None

    return STATUS_OK, struct.unpack(">I", core[2:6])[0]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Слепое извлечение и верификация ЦВЗ (DWT-DCT-QIM + Ed25519 + RS)."
    )
    p.add_argument("-i", "--input", required=True, help="анализируемое изображение")
    p.add_argument("-k", "--key", required=True, help="открытый ключ Ed25519 (PEM)")
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="шаг квантования QIM (должен совпадать с embedder)")
    p.add_argument("--no-rotate", action="store_true",
                   help="не перебирать повороты 90/180/270, если метка не найдена")
    p.add_argument("--verbose", action="store_true",
                   help="показывать результат каждой попытки (по ориентациям)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.delta <= 0:
        print("[FAIL] Параметр --delta должен быть положительным.")
        return 2

    try:
        public_key = load_public_key(args.key)
    except Exception as exc:
        print("[FAIL] Ошибка ключа: %s" % exc)
        return 2

    try:
        img = load_image(args.input)
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 2

    orientations = ROTATIONS if not args.no_rotate else ROTATIONS[:1]

    best_status = STATUS_RS
    for angle, flag in orientations:
        work = img if flag is None else cv2.rotate(img, flag)
        try:
            status, id_value = attempt_withdraw(work, args.delta, public_key)
        except ValueError as exc:
            print("[FAIL] %s" % exc)
            return 2

        if args.verbose:
            label = "исходная" if angle == 0 else "поворот %d°" % angle
            print("[i] %-12s -> %s" % (label, status))

        if status == STATUS_OK:
            if angle == 0:
                print("[SUCCESS] Изображение подтверждено. ID: %d" % id_value)
            else:
                print("[SUCCESS] Изображение подтверждено. ID: %d "
                      "(метка найдена при повороте %d°)" % (id_value, angle))
            return 0

        if _STATUS_RANK[status] > _STATUS_RANK[best_status]:
            best_status = status

    if best_status == STATUS_SIG:
        print("[FAIL] Подпись недействительна")
    else:
        print("[FAIL] Метка разрушена или отсутствует")
    return 1


if __name__ == "__main__":
    sys.exit(main())
