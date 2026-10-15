#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""artifact_demo.py — демонстратор визуальных артефактов стеганографии.

Намеренно «перегружает» пространственные и частотные области изображения
случайным шумом, эмулируя избыточное внедрение данных тремя методами:

  * LSB  (пространственный домен, RGB)   — lsb_1bit / lsb_4bit / lsb_7bit
  * DCT  (8x8, scipy)                    — dct_low_freq / dct_mid_freq / dct_high_freq
  * DWT  (Haar, 1 уровень, pywt)         — dwt_LL / dwt_HL_LH

Частотные методы работают в канале яркости (grayscale или Y из YUV) и
сохраняются двумя наборами: grayscale и цветным (изменённый Y возвращается
в исходные U/V). Пространственный LSB применяется к RGB.

Результаты пишутся рядом с исходным файлом (или в --outdir) по шаблону
    <имя>_<метод>[_<представление>].<ext>
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn

# --------------------------------------------------------------------------
# Параметры по умолчанию
# --------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_AMP_DCT = 150.0
DEFAULT_AMP_DWT = 80.0
BLOCK = 8

# Метод LSB -> число младших битов, которые заменяются шумом
LSB_BITS = {
    "lsb_1bit": 1,
    "lsb_4bit": 4,
    "lsb_7bit": 7,
}

# Метод DCT -> предикат выбора коэффициентов в блоке 8x8
#   low : верхний левый угол i<3, j<3 (без DC)
#   mid : главная диагональ i==j
#   high: нижний правый угол i>=5, j>=5
DCT_REGIONS = {
    "dct_low_freq": lambda i, j: i < 3 and j < 3 and not (i == 0 and j == 0),
    "dct_mid_freq": lambda i, j: i == j and i >= 1,
    "dct_high_freq": lambda i, j: i >= 5 and j >= 5,
}

# Метод DWT -> какие поддиапазоны зашумляются
DWT_SUBBANDS = {
    "dwt_LL": "LL",
    "dwt_HL_LH": "HL_LH",
}

ALL_METHODS = (
    list(LSB_BITS) + list(DCT_REGIONS) + list(DWT_SUBBANDS)
)


# --------------------------------------------------------------------------
# Чтение/запись (устойчиво к путям в UTF-8, напр. кириллице)
# --------------------------------------------------------------------------
def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError("не удалось прочитать файл: %s" % path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # всегда BGR, 3 канала
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
# Генератор артефактов
# --------------------------------------------------------------------------
class ArtifactGenerator:
    def __init__(self, path, seed=DEFAULT_SEED, ext="png",
                 amp_dct=DEFAULT_AMP_DCT, amp_dwt=DEFAULT_AMP_DWT, outdir=None):
        self.path = path
        self.img = load_image(path)
        self.stem = os.path.splitext(os.path.basename(path))[0]
        self.seed = int(seed)
        self.ext = ext.lstrip(".").lower()
        self.amp_dct = float(amp_dct)
        self.amp_dwt = float(amp_dwt)
        self.outdir = outdir or os.path.dirname(os.path.abspath(path))
        os.makedirs(self.outdir, exist_ok=True)

    # --- служебное ---------------------------------------------------------
    def _out_path(self, method, representation=None):
        name = "%s_%s" % (self.stem, method)
        if representation:
            name += "_" + representation
        return os.path.join(self.outdir, "%s.%s" % (name, self.ext))

    def _rng(self):
        return np.random.default_rng(self.seed)

    # --- 3.1 LSB (пространственный домен, RGB) -----------------------------
    def lsb(self, method):
        n = LSB_BITS[method]
        rng = self._rng()

        keep_mask = (0xFF << n) & 0xFF
        kept = self.img & np.uint8(keep_mask)
        noise = rng.integers(0, 1 << n, size=self.img.shape, dtype=np.uint8)
        out = np.bitwise_or(kept, noise).astype(np.uint8)

        path = self._out_path(method)
        save_image(path, out)
        return [path]

    # --- 3.2 DCT (8x8, канал яркости) --------------------------------------
    @staticmethod
    def _apply_dct(chan, region, amp, rng):
        h, w = chan.shape
        pad_h = (-h) % BLOCK
        pad_w = (-w) % BLOCK
        padded = np.pad(chan, ((0, pad_h), (0, pad_w)), mode="edge")
        H, W = padded.shape

        coords = [(i, j) for i in range(BLOCK) for j in range(BLOCK) if region(i, j)]
        out = padded.copy()
        for r in range(0, H, BLOCK):
            for c in range(0, W, BLOCK):
                block = padded[r:r + BLOCK, c:c + BLOCK]
                bd = dctn(block, type=2, norm="ortho")
                for (i, j) in coords:
                    bd[i, j] += rng.uniform(-amp, amp)
                out[r:r + BLOCK, c:c + BLOCK] = idctn(bd, type=2, norm="ortho")
        return np.clip(np.round(out[:h, :w]), 0, 255).astype(np.uint8)

    def dct_band(self, method):
        region = DCT_REGIONS[method]
        amp = self.amp_dct
        produced = []

        # grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_out = self._apply_dct(gray, region, amp, self._rng())
        p = self._out_path(method, "gray")
        save_image(p, gray_out)
        produced.append(p)

        # цветной: изменяется только Y, U/V не трогаются
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0].astype(np.float64)
        yuv[:, :, 0] = self._apply_dct(y, region, amp, self._rng())
        color_out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        p = self._out_path(method, "rgb")
        save_image(p, color_out)
        produced.append(p)

        return produced

    # --- 3.3 DWT (Haar, 1 уровень, канал яркости) --------------------------
    @staticmethod
    def _apply_dwt(chan, which, amp, rng):
        cA, (cH, cV, cD) = pywt.dwt2(chan, "haar", mode="periodization")
        if which == "LL":
            cA = cA + rng.uniform(-amp, amp, size=cA.shape)
        elif which == "HL_LH":
            cH = cH + rng.uniform(-amp, amp, size=cH.shape)
            cV = cV + rng.uniform(-amp, amp, size=cV.shape)
        rec = pywt.idwt2((cA, (cH, cV, cD)), "haar", mode="periodization")
        rec = rec[:chan.shape[0], :chan.shape[1]]
        return np.clip(np.round(rec), 0, 255).astype(np.uint8)

    def dwt_band(self, method):
        which = DWT_SUBBANDS[method]
        amp = self.amp_dwt
        produced = []

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_out = self._apply_dwt(gray, which, amp, self._rng())
        p = self._out_path(method, "gray")
        save_image(p, gray_out)
        produced.append(p)

        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0].astype(np.float64)
        yuv[:, :, 0] = self._apply_dwt(y, which, amp, self._rng())
        color_out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        p = self._out_path(method, "rgb")
        save_image(p, color_out)
        produced.append(p)

        return produced

    # --- диспетчер ---------------------------------------------------------
    def run_one(self, method):
        if method in LSB_BITS:
            return self.lsb(method)
        if method in DCT_REGIONS:
            return self.dct_band(method)
        if method in DWT_SUBBANDS:
            return self.dwt_band(method)
        raise ValueError("неизвестный метод: %s" % method)

    def run_all(self, methods=None):
        produced = []
        for method in (methods if methods is not None else ALL_METHODS):
            produced.extend(self.run_one(method))
        return produced


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Демонстрация артефактов перегрузки ЦВЗ (LSB / DCT / DWT)."
    )
    p.add_argument("-i", "--input", required=True, help="исходное изображение (PNG/JPG)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed ГСЧ (по умолчанию 42)")
    p.add_argument("--methods", help="список методов через запятую (по умолчанию все)")
    p.add_argument("--amp-dct", type=float, default=DEFAULT_AMP_DCT, help="амплитуда шума DCT")
    p.add_argument("--amp-dwt", type=float, default=DEFAULT_AMP_DWT, help="амплитуда шума DWT")
    p.add_argument("--ext", default="png", help="расширение выходных файлов (по умолчанию png)")
    p.add_argument("--outdir", help="каталог для результатов (по умолчанию рядом с исходником)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in ALL_METHODS]
        if unknown:
            print("[FAIL] Неизвестные методы: %s" % ", ".join(unknown))
            print("       Доступно: %s" % ", ".join(ALL_METHODS))
            return 2

    try:
        gen = ArtifactGenerator(
            args.input,
            seed=args.seed,
            ext=args.ext,
            amp_dct=args.amp_dct,
            amp_dwt=args.amp_dwt,
            outdir=args.outdir,
        )
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 2

    try:
        produced = gen.run_all(methods)
    except Exception as exc:
        print("[FAIL] %s" % exc)
        return 1

    print("[OK] Сгенерировано файлов: %d" % len(produced))
    print("     Каталог: %s" % gen.outdir)
    for path in produced:
        print("     - %s" % os.path.basename(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
