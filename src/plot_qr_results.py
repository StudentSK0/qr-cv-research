import csv
import matplotlib.pyplot as plt
import os

# Абсолютный путь к корню проекта
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(ROOT, "outputs", "qr_experiment_results.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# Чтение CSV
scales = []
times = []
success = []

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scales.append(float(row["scale"]))
        times.append(float(row["avg_time_ms"]))
        success.append(float(row["success_rate"]))

# График 1
plt.figure(figsize=(8, 4))
plt.plot(scales, times, marker='o')
plt.gca().invert_xaxis()
plt.title("QR Decoding Time vs Scale")
plt.xlabel("Scale")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.tight_layout()

output_file_1 = os.path.join(OUTPUT_DIR, "qr_time_vs_scale.png")
plt.savefig(output_file_1, dpi=300)
print(f"Saved: {output_file_1}")
plt.close()

# График 2
plt.figure(figsize=(8, 4))
plt.plot(scales, success, marker='o')
plt.gca().invert_xaxis()
plt.title("QR Decoding Success vs Scale")
plt.xlabel("Scale")
plt.ylabel("Success Rate")
plt.grid(True)
plt.tight_layout()

output_file_2 = os.path.join(OUTPUT_DIR, "qr_success_vs_scale.png")
plt.savefig(output_file_2, dpi=300)
print(f"Saved: {output_file_2}")
plt.close()
