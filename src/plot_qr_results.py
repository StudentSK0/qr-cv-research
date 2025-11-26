import csv
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

# Абсолютный путь к корню проекта
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(ROOT, "outputs", "qr_experiment_results.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# -------------------------------------------
# Чтение CSV и агрегация по масштабу
# -------------------------------------------
scale_to_times = defaultdict(list)
scale_to_success = defaultdict(list)

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scale = float(row["scale"])
        avg_time_ms = float(row["avg_time_ms"])
        success_rate = float(row["success_rate"])

        scale_to_times[scale].append(avg_time_ms)
        scale_to_success[scale].append(success_rate)

# -------------------------------------------
# Усреднение данных
# -------------------------------------------
scales = sorted(scale_to_times.keys(), reverse=True)  # от 1.0 к меньшим
mean_times = [np.mean(scale_to_times[s]) for s in scales]
mean_success = [np.mean(scale_to_success[s]) for s in scales]

# -------------------------------------------
# Построение комбинированного графика
# -------------------------------------------
plt.figure(figsize=(10, 5))

# Время (левая ось)
ax1 = plt.gca()
ax1.plot(scales, mean_times, marker='o', color="tab:blue", label="Mean Decoding Time (ms)")
ax1.set_xlabel("Scale")
ax1.set_ylabel("Time (ms)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.invert_xaxis()
ax1.grid(True, axis="both", linestyle="--", alpha=0.5)

# Успешность (правая ось)
ax2 = ax1.twinx()
ax2.plot(scales, mean_success, marker='s', color="tab:red", label="Mean Success Rate")
ax2.set_ylabel("Success Rate", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Заголовок
plt.title("Average QR Decoding Performance vs Scale (All QR Codes)")

# Общая легенда
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="lower center")

plt.tight_layout()

# Сохранение
output_file = os.path.join(OUTPUT_DIR, "qr_combined_mean_plot.png")
plt.savefig(output_file, dpi=300)
print(f"Saved combined mean plot: {output_file}")

plt.close()
