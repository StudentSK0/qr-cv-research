import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from InquirerPy import inquirer
from pyzxing import BarCodeReader

# Количество итераций декодирования для усреднения
DECODE_ITERATIONS = 1

# >>> ВАЖНО: шаг бининга (квантизации) размера модуля в пикселях
# 1 -> просто округление до целого
# 2/3/5 -> группировка по корзинам шага
MODULE_BIN_STEP_PX = 2

reader = BarCodeReader()  # Инициализация ZXing один раз


def quantize_module_size_px(module_size_px: float, step_px: int = MODULE_BIN_STEP_PX) -> int:
    """
    Приводит module_size к целому числу пикселей и выполняет биннинг по step_px.

    rounded = round(module_size_px)
    q = round(rounded / step_px) * step_px  (если step_px > 1)

    Пример:
      module=3.7 -> rounded=4
      step=2 -> q=4
      step=5 -> q=5 или 0/10 (здесь: ближайшая корзина)
    """
    if module_size_px is None:
        return None

    if step_px < 1:
        step_px = 1

    rounded = int(np.rint(module_size_px))
    if step_px == 1:
        return rounded

    q = int(np.rint(rounded / step_px) * step_px)
    return q


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

    return decoded, float(elapsed)


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
    if not datasets_path.exists():
        print("Папка datasets не найдена!")
        return None

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
    """Обрабатывает датасет по аналогии с OpenCV-версией (с биннингом module_size)."""
    results = []

    base = project_root / "datasets" / dataset_name
    images = base / "images" / "QR_CODE"
    markup = base / "markup" / "QR_CODE"

    print(f"\nZXing: обработка датасета '{dataset_name}'")
    print(f"Изображения: {images}")
    print(f"Разметка:    {markup}")
    print(f"Квантизация размера модуля: шаг = {MODULE_BIN_STEP_PX} px")

    if not images.exists() or not markup.exists():
        print("Ошибка: отсутствуют images/QR_CODE или markup/QR_CODE")
        return results

    image_files = []
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
        image_files.extend(images.glob(f"*{ext}"))

    processed = 0
    skipped_no_markup = 0
    skipped_bad_module = 0

    for img in image_files:
        meta = markup / f"{img.name}.json"
        if not meta.exists():
            skipped_no_markup += 1
            continue

        try:
            with open(meta, "r", encoding="utf-8") as f:
                markup_data = json.load(f)
        except Exception:
            continue

        module_size_raw = get_module_size(markup_data)
        if module_size_raw is None:
            skipped_bad_module += 1
            continue

        # >>> ключевое изменение: целые пиксели + биннинг по шагу
        module_size_px = quantize_module_size_px(module_size_raw, MODULE_BIN_STEP_PX)
        if module_size_px is None:
            skipped_bad_module += 1
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
            "module_size": int(module_size_px),          # <<< целое + бининг
            "module_size_raw": float(module_size_raw),   # <<< сохраняем исходное
            "time": float(decode_time),
            "accuracy": float(avg_accuracy),
            "decoded": decoded_best,
            "expected": expected,
            "image_path": str(img)
        })

        processed += 1
        if processed % 100 == 0:
            print(f"Обработано: {processed}")

    print(f"Всего обработано: {processed}")
    if skipped_no_markup:
        print(f"Пропущено без разметки: {skipped_no_markup}")
    if skipped_bad_module:
        print(f"Пропущено из-за module_size: {skipped_bad_module}")

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
