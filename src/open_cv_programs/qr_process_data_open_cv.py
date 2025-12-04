import os
import json
import cv2
import numpy as np
from pathlib import Path
import time
from collections import Counter
from InquirerPy import inquirer

# Количество итераций декодирования для усреднения
DECODE_ITERATIONS = 3


def get_module_size(markup_data):
    """Извлекает средний размер модуля из файла разметки"""
    try:
        module_size = markup_data['props']['barcode']['module_size']
        # Преобразуем 2 диапазона [x_min, x_max], [y_min, y_max] в плоский массив
        all_values = []
        for pair in module_size:
            all_values.extend(pair)
        return np.mean(all_values)
    except (KeyError, TypeError):
        return None


def decode_qr_code_single(image_path):
    """Декодирует QR один раз и возвращает (decoded_value, time_seconds)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    detector = cv2.QRCodeDetector()

    # Сначала пытаемся через detectAndDecodeMulti
    start_time = time.time()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
    elapsed_time = time.time() - start_time

    if retval:
        # decoded_info — кортеж/список строк или одна строка
        if isinstance(decoded_info, (tuple, list)) and len(decoded_info) > 0:
            first_value = decoded_info[0]
            decoded_value = str(first_value) if first_value else ''
        elif isinstance(decoded_info, str) and decoded_info:
            decoded_value = decoded_info
        else:
            decoded_value = ''
        return decoded_value, elapsed_time
    else:
        # Если не удалось, используем обычный detectAndDecode
        start_time = time.time()
        retval, decoded_info, points = detector.detectAndDecode(img)
        elapsed_time = time.time() - start_time

        if retval:
            if decoded_info is not None:
                decoded_str = str(decoded_info)
                if decoded_str:
                    return decoded_str, elapsed_time
            return '', elapsed_time
        else:
            return '', elapsed_time


def decode_qr_code(image_path, iterations=DECODE_ITERATIONS):
    """
    Повторное декодирование для оценки точности:
    - возвращает список decoded_values (для accuracy),
    - минимальное время декодирования (в секундах).
    """
    decoded_values = []
    times = []

    for _ in range(iterations):
        decoded_value, decode_time = decode_qr_code_single(image_path)
        if decode_time is not None:
            decoded_values.append(decoded_value)
            times.append(decode_time)

    if not times:
        return None, None

    return decoded_values, np.min(times)


def select_dataset(project_root: Path) -> str:
    """
    Выбор папки внутри project_root/datasets с помощью стрелок.
    """
    datasets_path = project_root / "datasets"

    if not datasets_path.exists():
        print("Папка datasets не найдена!")
        return None

    dataset_list = [d.name for d in datasets_path.iterdir() if d.is_dir()]
    if not dataset_list:
        print("Нет доступных датасетов в папке datasets")
        return None

    dataset_name = inquirer.select(
        message="Выберите датасет стрелками и нажмите Enter:",
        choices=dataset_list,
        default=dataset_list[0],
    ).execute()

    return dataset_name


def process_dataset(project_root: Path, dataset_name: str):
    """
    Обработка всех изображений указанного датасета в project_root/datasets/<dataset_name>.
    """
    results = []
    ideal_path = project_root / 'datasets' / dataset_name

    images_path = ideal_path / 'images' / 'QR_CODE'
    markup_path = ideal_path / 'markup' / 'QR_CODE'

    if not images_path.exists() or not markup_path.exists():
        print(f"images/QR_CODE или markup/QR_CODE отсутствует: {ideal_path}")
        return results

    print(f"\nОбработка датасета: {dataset_name}")
    print(f"   Путь к изображениям: {images_path}")
    print(f"   Путь к разметке:    {markup_path}")

    # Сбор всех картинок
    image_files = []
    for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
        image_files.extend(images_path.glob(f"*{ext}"))

    processed = 0
    for image_file in image_files:
        markup_file = markup_path / f"{image_file.name}.json"
        if not markup_file.exists():
            continue

        try:
            with open(markup_file, "r", encoding="utf-8") as f:
                markup_data = json.load(f)
        except Exception as e:
            print(f"Ошибка чтения разметки {markup_file}: {e}")
            continue

        module_size = get_module_size(markup_data)
        if module_size is None:
            continue

        try:
            expected_value = str(markup_data['props']['barcode'].get('value', ''))
        except KeyError:
            expected_value = ''

        decoded_values, decode_time = decode_qr_code(image_file)
        if decode_time is None:
            continue

        # Оценка точности по совпадению decoded vs expected
        accuracies = [
            1.0 if str(d) == expected_value else 0.0
            for d in decoded_values
        ]
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0

        most_common_decoded = (
            Counter(decoded_values).most_common(1)[0][0]
            if decoded_values else ''
        )

        results.append({
            "dataset": dataset_name,
            "module_size": float(module_size),
            "time": float(decode_time),   # в секундах, как в исходной версии
            "accuracy": float(avg_accuracy),
            "decoded": most_common_decoded,
            "expected": expected_value,
            "image_path": str(image_file),
        })

        processed += 1
        if processed % 100 == 0:
            print(f"На текущий момент обработано: {processed} изображений")

    print(f"Всего обработано в {dataset_name}: {processed}")
    return results


def main():
    # qr_process_data_open_cv.py находится в project_root/src
    # поднимаемся на уровень вверх → project_root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    print("\nНачало обработки данных")
    print(f"Корень проекта: {project_root}")

    dataset_name = select_dataset(project_root)
    if not dataset_name:
        return

    results = process_dataset(project_root, dataset_name)
    if not results:
        print("Нет данных для обработки")
        return

    # Сохраняем JSON в project_root/outputs/
    output_dir = project_root / "outputs" / "open_cv_json_and_graphics"
    output_dir.mkdir(exist_ok=True)

    output_json = output_dir / f"qr_experiment_data_{dataset_name}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены в: {output_json}")
    print("Далее можно запустить src/open_cv_programs/qr_plot_results_open_cv.py для построения графика")


if __name__ == "__main__":
    main()
