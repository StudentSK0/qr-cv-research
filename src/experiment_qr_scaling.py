import cv2
import time
import numpy as np
import csv
import os

# -------------------------------------------
# Пути проекта
# -------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
CSV_PATH = os.path.join(OUTPUT_DIR, "qr_experiment_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------
# Настройки эксперимента
# -------------------------------------------
NUM_ITERS = 20
MIN_SCALE = 0.05
MAX_SCALE = 1.0
NUM_SCALES = 20

scales = np.linspace(MAX_SCALE, MIN_SCALE, NUM_SCALES)
detector = cv2.QRCodeDetector()

# -------------------------------------------
# Получаем список всех изображений в data/
# -------------------------------------------
qr_files = [
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not qr_files:
    raise RuntimeError(f"No QR images found in {DATA_DIR}")

print("QR files found:", qr_files)

# -------------------------------------------
# Основной цикл: обрабатываем каждый QR-файл
# -------------------------------------------
results = []

for filename in qr_files:
    image_path = os.path.join(DATA_DIR, filename)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: cannot read {filename}, skipping...")
        continue

    print(f"\nProcessing {filename}...")

    for s in scales:
        scaled = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        times = []
        decoded_ok = 0

        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            data, bbox, _ = detector.detectAndDecode(scaled)
            t1 = time.perf_counter()

            times.append(t1 - t0)
            if data:
                decoded_ok += 1

        avg_time_ms = (sum(times) / NUM_ITERS) * 1000
        success_rate = decoded_ok / NUM_ITERS

        results.append((filename, s, avg_time_ms, success_rate))

        print(f"{filename}: scale={s:.2f}, time={avg_time_ms:.2f} ms, success={success_rate:.2f}")

# -------------------------------------------
# Сохранение общего CSV
# -------------------------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "scale", "avg_time_ms", "success_rate"])
    writer.writerows(results)

print(f"\nCSV saved to: {CSV_PATH}")
