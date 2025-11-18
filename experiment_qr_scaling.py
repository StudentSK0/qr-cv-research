import cv2
import time
import numpy as np
import csv

# -------------------------------------------
# Настройки эксперимента
# -------------------------------------------
IMAGE_PATH = "./pics/qr-code.png"
CSV_PATH = "qr_experiment_results.csv"

NUM_ITERS = 20      # число повторов для усреднения
MIN_SCALE = 0.05     # минимальный масштаб
MAX_SCALE = 1.0      # максимальный масштаб
NUM_SCALES = 20      # количество измерений по шкале

# -------------------------------------------
# Загрузка QR и проверка
# -------------------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Файл {IMAGE_PATH} не найден")

detector = cv2.QRCodeDetector()

scales = np.linspace(MAX_SCALE, MIN_SCALE, NUM_SCALES)
results = []

# -------------------------------------------
# Основной цикл эксперимента
# -------------------------------------------
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

    results.append((s, avg_time_ms, success_rate))
    print(f"scale={s:.2f}, time={avg_time_ms:.2f} ms, success={success_rate:.2f}")

# -------------------------------------------
# Сохранение CSV
# -------------------------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scale", "avg_time_ms", "success_rate"])
    writer.writerows(results)

print(f"\nCSV сохранён: {CSV_PATH}")
