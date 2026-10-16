#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark.py — эмулятор атак и стресс-тестирование каскада DWT-DCT-QIM.

Скрипт:
  1. Берёт набор оригинальных изображений из каталога.
  2. Встраивает в каждое метку (ID + подпись Ed25519) — через embedder.py.
  3. Применяет набор атак (JPEG / AWGN / rescale / crop / rotate) с заданным
     шагом интенсивности.
  4. Для каждой итерации считает PSNR, BER (по сырым 800 битам до RS) и
     статус успешного извлечения (RS + подпись Ed25519 + совпадение ID).
  5. Пишет сводный отчёт в CSV и/или JSON, при желании — графики (matplotlib).

Логика встраивания/извлечения импортируется напрямую из embedder.py и
detector.py (без вызова через subprocess).
"""

import argparse
import csv
import json
import os
import struct
import sys

import cv2
import numpy as np

# --- импорт логики ЦВЗ из соседних модулей --------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embedder as E      # noqa: E402
import detector as D      # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                                   # pragma: no cover
    HAVE_MPL = False


# --------------------------------------------------------------------------
# Параметры по умолчанию
# --------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DEFAULT_DELTA = 30
DEFAULT_SEED = 42
DEFAULT_ID = 12345

JPEG_QUALITIES = [95, 80, 60, 40, 20, 10]
GAUSSIAN_SIGMAS = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
RESCALE_FACTORS = [0.9, 0.7, 0.5]
CROP_PIXELS = [1, 2, 5, 10]
ROTATION_ANGLES = [90, 180, 270]


# --------------------------------------------------------------------------
# Атаки
# --------------------------------------------------------------------------
def attack_jpeg(img, quality, rng=None):
    """Сжатие с потерями: PNG -> JPEG с заданным quality."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("ошибка JPEG-кодирования")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_gaussian(img, sigma, rng=None):
    """Аддитивный белый гауссов шум (AWGN)."""
    noise = rng.normal(0.0, float(sigma), img.shape)
    out = img.astype(np.float64) + noise
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def attack_rescale(img, factor, rng=None):
    """Downscale -> upscale бикубической интерполяцией."""
    h, w = img.shape[:2]
    nw = max(1, int(round(w * factor)))
    nh = max(1, int(round(h * factor)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def attack_crop(img, px, rng=None):
    """Обрезка px пикселей с левого верхнего края (рассинхронизация сетки)."""
    px = int(px)
    return img[px:, px:].copy()


def attack_rotate(img, angle, rng=None):
    """Ортогональный поворот на 90/180/270 градусов."""
    flags = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(img, flags[int(angle)])


# (имя, набор параметров, функция, подпись параметра)
ATTACKS = [
    ("jpeg", JPEG_QUALITIES, attack_jpeg, "quality"),
    ("gaussian", GAUSSIAN_SIGMAS, attack_gaussian, "sigma"),
    ("rescale", RESCALE_FACTORS, attack_rescale, "factor"),
    ("crop", CROP_PIXELS, attack_crop, "px"),
    ("rotate", ROTATION_ANGLES, attack_rotate, "angle"),
]


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------
def psnr(ref, test):
    """PSNR между оригиналом и изображением после атаки (дБ).

    Если размеры различаются (атака crop), сравнение идёт по наибольшей
    общей (выровненной по левому верхнему углу) области.
    """
    a = ref.astype(np.float64)
    b = test.astype(np.float64)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def evaluate(img, bits_true, public_key, embedded_id, delta, rotate_search=True):
    """Попытка извлечь метку из атакованного изображения.

    Возвращает (angle | None, raw_bits | None, success_bool).
    raw_bits — сырой извлечённый поток (для расчёта BER). Если метка найдена,
    это биты с удачной ориентации; иначе — с исходной (0°).
    """
    orientations = D.ROTATIONS if rotate_search else D.ROTATIONS[:1]
    fallback = None

    for angle, flag in orientations:
        work = img if flag is None else cv2.rotate(img, flag)
        try:
            raw = D.extract_bits(work, delta)
        except ValueError:
            continue
        if fallback is None:
            fallback = raw
        status, id_value, _ = D.verify_bits(raw, public_key)
        if status == D.STATUS_OK and id_value == embedded_id:
            return angle, raw, True

    return None, fallback, False


# --------------------------------------------------------------------------
# Вывод отчётов
# --------------------------------------------------------------------------
FIELDS = ["image", "attack", "param", "psnr", "ber", "success", "angle", "id"]


def _fmt_num(x, nd=3):
    if x is None:
        return ""
    if isinstance(x, float) and (x == float("inf")):
        return "inf"
    return round(float(x), nd)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "image": r["image"],
                "attack": r["attack"],
                "param": r["param"],
                "psnr": _fmt_num(r["psnr"]),
                "ber": _fmt_num(r["ber"], 4),
                "success": int(bool(r["success"])),
                "angle": "" if r["angle"] is None else r["angle"],
                "id": "" if r["id"] is None else r["id"],
            })


def write_json(path, rows):
    def clean(r):
        out = dict(r)
        if out["psnr"] is not None and out["psnr"] == float("inf"):
            out["psnr"] = None
        return out

    with open(path, "w", encoding="utf-8") as f:
        json.dump([clean(r) for r in rows], f, ensure_ascii=False, indent=2)


def summarize(rows):
    """Печать сводки: доля успешных извлечений и средний BER по типам атак."""
    print("\n%-10s %-8s %8s %10s" % ("attack", "param", "success", "mean BER"))
    print("-" * 40)
    for name, params, _fn, _pn in ATTACKS:
        for p in params:
            subset = [r for r in rows if r["attack"] == name and r["param"] == p]
            if not subset:
                continue
            succ = np.mean([bool(r["success"]) for r in subset])
            bers = [r["ber"] for r in subset if r["ber"] is not None]
            mean_ber = float(np.mean(bers)) if bers else float("nan")
            print("%-10s %-8s %7.0f%% %10.4f" % (name, p, 100.0 * succ, mean_ber))


def make_plots(rows, outdir):
    if not HAVE_MPL:
        print("[warn] matplotlib недоступен — графики пропущены")
        return []
    os.makedirs(outdir, exist_ok=True)
    produced = []
    for name, params, _fn, _pn in ATTACKS:
        xs, succ_rates, mean_bers = [], [], []
        for p in params:
            subset = [r for r in rows if r["attack"] == name and r["param"] == p]
            if not subset:
                continue
            xs.append(p)
            succ_rates.append(100.0 * np.mean([bool(r["success"]) for r in subset]))
            bers = [r["ber"] for r in subset if r["ber"] is not None]
            mean_bers.append(float(np.mean(bers)) if bers else 0.0)
        if not xs:
            continue

        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(xs, succ_rates, "o-", color="tab:green", label="Success rate")
        ax1.set_xlabel(_pn)
        ax1.set_ylabel("Success rate, %", color="tab:green")
        ax1.set_ylim(-2, 102)
        ax1.tick_params(axis="y", labelcolor="tab:green")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs, mean_bers, "s--", color="tab:red", label="Mean BER")
        ax2.set_ylabel("Mean BER", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title("Атака: %s" % name)
        fig.tight_layout()
        path = os.path.join(outdir, "attack_%s.png" % name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        produced.append(path)
    return produced


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Стресс-тестирование ЦВЗ: атаки + метрики PSNR/BER/success."
    )
    p.add_argument("-d", "--dir", required=True, help="каталог с оригинальными изображениями")
    p.add_argument("-k", "--key", help="закрытый ключ Ed25519 (PEM). Нет — генерируется временная пара")
    p.add_argument("-m", "--message", default=str(DEFAULT_ID), help="ID автора (по умолчанию %d)" % DEFAULT_ID)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="шаг квантования QIM (по умолчанию 30)")
    p.add_argument("-o", "--output", default="benchmark_report.csv",
                   help="отчёт: .csv или .json (по расширению)")
    p.add_argument("--json", dest="json_out", help="дополнительно сохранить JSON")
    p.add_argument("--plots", help="каталог для графиков (matplotlib)")
    p.add_argument("--attacks", help="подмножество атак через запятую (по умолчанию все)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed ГСЧ")
    p.add_argument("--limit", type=int, help="ограничить число изображений")
    p.add_argument("--no-rotate", action="store_true", help="не использовать перебор поворотов детектором")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.delta <= 0:
        print("[FAIL] --delta должен быть положительным.")
        return 2

    # --- ключи и эталонный битовый поток ---
    try:
        if args.key:
            private_key = E.load_or_generate_key(args.key)[0]
        else:
            private_key, paths = E.load_or_generate_key(None)
            print("[i] Сгенерирована временная пара ключей: %s" % paths[1])
    except Exception as exc:
        print("[FAIL] Ошибка ключа: %s" % exc)
        return 2
    public_key = private_key.public_key()

    bits_true, core = E.build_bits(args.message, private_key)
    embedded_id = struct.unpack(">I", core[2:6])[0]

    # --- отбор атак ---
    selected = ATTACKS
    if args.attacks:
        wanted = {a.strip() for a in args.attacks.split(",") if a.strip()}
        selected = [a for a in ATTACKS if a[0] in wanted]
        unknown = wanted - {a[0] for a in ATTACKS}
        if unknown:
            print("[FAIL] Неизвестные атаки: %s" % ", ".join(sorted(unknown)))
            print("       Доступно: %s" % ", ".join(a[0] for a in ATTACKS))
            return 2

    # --- изображения ---
    if not os.path.isdir(args.dir):
        print("[FAIL] Не найден каталог: %s" % args.dir)
        return 2
    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(IMAGE_EXTS))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("[FAIL] В каталоге нет изображений поддерживаемых форматов.")
        return 2

    print("[i] Изображений: %d | Атак: %d | ID: %d | delta=%g"
          % (len(files), sum(len(a[1]) for a in selected), embedded_id, args.delta))

    rows = []
    for idx, fname in enumerate(files):
        path = os.path.join(args.dir, fname)
        try:
            img = E.load_image(path)
        except Exception as exc:
            print("[skip] %s: %s" % (fname, exc))
            continue
        try:
            marked = E.embed(img, bits_true, args.delta)
        except ValueError as exc:
            print("[skip] %s: %s" % (fname, exc))
            continue

        rng = np.random.default_rng([args.seed, idx])
        for name, params, fn, _pn in selected:
            for p in params:
                attacked = fn(marked, p, rng)
                p_attack = psnr(img, attacked)
                angle, raw, ok = evaluate(
                    attacked, bits_true, public_key, embedded_id,
                    args.delta, rotate_search=not args.no_rotate,
                )
                ber = None if raw is None else float(np.mean(raw != bits_true))
                rows.append({
                    "image": fname,
                    "attack": name,
                    "param": p,
                    "psnr": p_attack,
                    "ber": ber,
                    "success": bool(ok),
                    "angle": angle,
                    "id": embedded_id if ok else None,
                })
        print("[ok] %-30s обработано (%d итераций накоплено)" % (fname, len(rows)))

    # --- запись отчётов ---
    out = args.output
    if out.lower().endswith(".json"):
        write_json(out, rows)
    else:
        write_csv(out, rows)
    print("\n[i] Отчёт: %s" % out)

    if args.json_out:
        write_json(args.json_out, rows)
        print("[i] JSON: %s" % args.json_out)

    summarize(rows)

    if args.plots:
        produced = make_plots(rows, args.plots)
        if produced:
            print("\n[i] Графики (%d): %s" % (len(produced), args.plots))

    return 0


if __name__ == "__main__":
    sys.exit(main())
