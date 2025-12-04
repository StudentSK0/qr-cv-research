import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from InquirerPy import inquirer
from pyzxing import BarCodeReader

# Количество итераций декодирования для усреднения
DECODE_ITERATIONS = 1

reader = BarCodeReader()  # Инициализация ZXing один раз


def get_module_size(markup_data):
    """Извлекает средний размер модуля из файла разметки."""
    try:
        module_size = markup_data['props']['barcode']['module_size']
        all_values = []
        for pair in module_size:
            all_values.extend(pair)
        return float(np.mean(all_values))
    except Exception:
        return None


def decode_qr_code_single(image_path: Path):
    """Декодирует QR один раз (ZXing). Возвращает (decoded_string, time_sec)."""
    start = time.perf_counter()
    result = reader.decode(str(image_path))
    elapsed = time.perf_counter() - start

    decoded = ""
    if result and len(result) > 0:
        parsed = result[0].get("parsed", "")
        if isinstance(parsed, bytes):
            parsed = parsed.decode("utf-8", errors="ignore")
        decoded = str(parsed)

    return decoded, elapsed


def decode_qr_code(image_path: Path, iterations=DECODE_ITERATIONS):
    """Повторное декодирование для оценки стабильности-времени и точности."""
    decoded_values = []
    times = []

    for _ in range(iterations):
        decoded, t = decode_qr_code_single(image_path)
        decoded_values.append(decoded)
        times.append(t)

    if not times:
        return None, None

    return decoded_values, float(np.min(times))


def select_dataset(project_root: Path) -> str:
    """Выбор папки датасета из project_root/datasets"""
    datasets_path = project_root / "datasets"
    dataset_list = [d.name for d in datasets_path.iterdir() if d.is_dir()]

    if not dataset_list:
        print("Нет папок датасетов.")
        return None

    dataset_name = inquirer.select(
        message="Выберите датасет для ZXing:",
        choices=dataset_list,
        default=dataset_list[0]
    ).execute()

    return dataset_name


def process_dataset(project_root: Path, dataset_name: str):
    """Обрабатывает датасет по аналогии с OpenCV-версией."""
    results = []

    base = project_root / "datasets" / dataset_name
    images = base / "images" / "QR_CODE"
    markup = base / "markup" / "QR_CODE"

    print(f"\nZXing: обработка датасета '{dataset_name}'")
    print(f"Изображения: {images}")
    print(f"Разметка:    {markup}")

    if not images.exists() or not markup.exists():
        print("Ошибка: отсутствуют images/QR_CODE или markup/QR_CODE")
        return results

    image_files = []
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
        image_files.extend(images.glob(f"*{ext}"))

    processed = 0
    for img in image_files:
        meta = markup / f"{img.name}.json"
        if not meta.exists():
            continue

        try:
            with open(meta, "r", encoding="utf-8") as f:
                markup_data = json.load(f)
        except Exception:
            continue

        module_size = get_module_size(markup_data)
        if module_size is None:
            continue

        expected = str(markup_data['props']['barcode'].get('value', ""))

        decoded_values, decode_time = decode_qr_code(img)
        if decode_time is None:
            continue

        accuracies = [1.0 if d == expected else 0.0 for d in decoded_values]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        decoded_best = Counter(decoded_values).most_common(1)[0][0] if decoded_values else ""

        results.append({
            "dataset": dataset_name,
            "module_size": module_size,
            "time": decode_time,
            "accuracy": avg_accuracy,
            "decoded": decoded_best,
            "expected": expected,
            "image_path": str(img)
        })

        processed += 1
        if processed % 100 == 0:
            print(f"Обработано: {processed}")

    print(f"Всего обработано: {processed}")
    return results


def main():
    script = Path(__file__).resolve()
    project_root = script.parent.parent.parent

    print("\nЗапуск ZXing обработки")
    print(f"Корень проекта: {project_root}")

    dataset_name = select_dataset(project_root)
    if not dataset_name:
        return

    results = process_dataset(project_root, dataset_name)
    if not results:
        print("Нет данных для сохранения")
        return

    out_dir = project_root / "outputs" / "zxing_json_and_graphics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"qr_experiment_data_{dataset_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nJSON сохранён: {out_json}")
    print("Можно построить графики: qr_plot_results_zxing.py")


if __name__ == "__main__":
    main()
