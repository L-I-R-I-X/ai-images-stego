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
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Слепое извлечение и верификация ЦВЗ (DWT-DCT-QIM + Ed25519 + RS)."
    )
    p.add_argument("-i", "--input", required=True, help="анализируемое изображение")
    p.add_argument("-k", "--key", required=True, help="открытый ключ Ed25519 (PEM)")
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="шаг квантования QIM (должен совпадать с embedder)")
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

    try:
        bits = extract_bits(img, args.delta)
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 2

    code = np.packbits(bits).tobytes()

    try:
        payload = rs_decode(code)
    except reedsolo.ReedSolomonError:
        print("[FAIL] Метка разрушена или отсутствует")
        return 1
    except Exception as exc:
        print("[FAIL] Ошибка декодирования: %s" % exc)
        return 1

    if len(payload) != PAYLOAD_LEN:
        print("[FAIL] Метка разрушена или отсутствует")
        return 1

    core = payload[:CORE_LEN]
    signature = payload[CORE_LEN:CORE_LEN + SIG_LEN]

    if core[:2] != MAGIC:
        print("[FAIL] Метка разрушена или отсутствует")
        return 1

    try:
        public_key.verify(signature, core)
    except InvalidSignature:
        print("[FAIL] Подпись недействительна")
        return 1

    id_value = struct.unpack(">I", core[2:6])[0]
    print("[SUCCESS] Изображение подтверждено. ID: %d" % id_value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
