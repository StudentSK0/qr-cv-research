import csv
import matplotlib.pyplot as plt

CSV_PATH = "qr_experiment_results.csv"

# -------------------------------------------
# Чтение CSV
# -------------------------------------------
scales = []
times = []
success = []

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scales.append(float(row["scale"]))
        times.append(float(row["avg_time_ms"]))
        success.append(float(row["success_rate"]))

# -------------------------------------------
# График 1: Время считывания
# -------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(scales, times, marker='o')
plt.gca().invert_xaxis()
plt.title("Время считывания QR-кода vs Масштаб")
plt.xlabel("Масштаб (доля исходного размера)")
plt.ylabel("Время (мс)")
plt.grid(True)
plt.tight_layout()

# Сохранение
output_file_1 = "qr_time_vs_scale.png"
plt.savefig(output_file_1, dpi=300)
print(f"Сохранён график: {output_file_1}")

plt.close()

# -------------------------------------------
# График 2: Успешность считывания
# -------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(scales, success, marker='o')
plt.gca().invert_xaxis()
plt.title("Успешность декодирования vs Масштаб")
plt.xlabel("Масштаб")
plt.ylabel("Успешность (0–1)")
plt.grid(True)
plt.tight_layout()

# Сохранение
output_file_2 = "qr_success_vs_scale.png"
plt.savefig(output_file_2, dpi=300)
print(f"Сохранён график: {output_file_2}")

plt.close()
