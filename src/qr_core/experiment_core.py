# src/qr_core/experiment_core.py
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset_io import iter_qr_samples, read_json
from .module_size import get_module_size_raw, quantize_module_size_px
from .engines.base import QRDecoder


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str
    decode_iterations: int = 3
    module_bin_step_px: int = 1
    time_mode: str = "time_total_min"   # фиксируем выбранный режим
    progress_each: int = 100


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def run_experiment(
    project_root: Path,
    decoder: QRDecoder,
    cfg: ExperimentConfig,
) -> list[dict]:
    results: list[dict] = []

    processed = 0
    skipped_bad_markup = 0
    skipped_bad_module = 0

    for image_path, markup_path in iter_qr_samples(project_root, cfg.dataset_name):
        markup = read_json(markup_path)
        if not markup:
            skipped_bad_markup += 1
            continue

        module_raw = get_module_size_raw(markup)
        if module_raw is None:
            skipped_bad_module += 1
            continue

        module_px = quantize_module_size_px(module_raw, cfg.module_bin_step_px)
        if module_px is None:
            skipped_bad_module += 1
            continue

        expected = _safe_str(markup.get("props", {}).get("barcode", {}).get("value", ""))

        decoded_values: list[str] = []
        times: list[float] = []

        iters = int(cfg.decode_iterations)
        if iters < 1:
            iters = 1

        for _ in range(iters):
            t0 = time.perf_counter()
            decoded = decoder.decode(image_path)
            t1 = time.perf_counter()

            decoded = _safe_str(decoded)
            decoded_values.append(decoded)
            times.append(float(t1 - t0))

        if not times:
            continue

        time_min = float(np.min(times))
        acc_list = [1.0 if d.strip() else 0.0 for d in decoded_values]
        acc_mean = float(np.mean(acc_list)) if acc_list else 0.0
        decoded_best = Counter(decoded_values).most_common(1)[0][0] if decoded_values else ""

        results.append(
            {
                "dataset": cfg.dataset_name,
                "engine": decoder.name,
                "module_size": int(module_px),
                "module_size_raw": float(module_raw),
                "module_bin_step_px": int(cfg.module_bin_step_px),
                "time": time_min,
                "time_mode": cfg.time_mode,
                "accuracy": acc_mean,
                "decoded": decoded_best,
                "expected": expected,
                "image_path": str(image_path),
                "decode_iterations": int(iters),
            }
        )

        processed += 1
        if cfg.progress_each > 0 and processed % cfg.progress_each == 0:
            print(f"Обработано: {processed}")

    print(f"Готово. Обработано: {processed}")
    if skipped_bad_markup:
        print(f"Пропущено (не читается разметка): {skipped_bad_markup}")
    if skipped_bad_module:
        print(f"Пропущено (нет/битый module_size): {skipped_bad_module}")

    return results


def save_results_json(project_root: Path, engine_name: str, dataset_name: str, results: list[dict]) -> Path:
    out_dir = project_root / "outputs" / f"{engine_name}_json_and_graphics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"qr_experiment_data_{dataset_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return out_json
