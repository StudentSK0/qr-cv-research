from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .binning import quantize_module_size
from .dataset_io import iter_qr_samples
from .engines.base import BaseEngine, EngineError
from .markup import extract_expected_value, extract_module_size_px, read_markup


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str
    decode_iterations: int = 3
    module_bin_step_px: int = 2
    time_mode: str = "time_total_min"


@dataclass(frozen=True)
class SampleResult:
    dataset: str
    engine: str
    module_size: int
    module_size_raw: float
    module_bin_step_px: int
    time_total_min_sec: float
    time_mode: str
    accuracy: float
    decoded: str
    expected: str
    image_path: str
    decode_iterations: int


@dataclass(frozen=True)
class ExperimentSummary:
    processed: int
    skipped_no_markup: int
    skipped_bad_module: int
    skipped_image_read: int


@dataclass(frozen=True)
class ProgressState:
    seen: int
    processed: int
    skipped_no_markup: int
    skipped_bad_module: int
    skipped_image_read: int


def run_experiment(
    project_root: Path,
    engine: BaseEngine,
    cfg: ExperimentConfig,
    progress_cb: Callable[[ProgressState], None] | None = None,
) -> tuple[list[SampleResult], ExperimentSummary]:
    results: list[SampleResult] = []

    processed = 0
    skipped_no_markup = 0
    skipped_bad_module = 0
    skipped_image_read = 0
    seen = 0

    iterations = max(1, int(cfg.decode_iterations))

    for image_path, markup_path in iter_qr_samples(project_root, cfg.dataset_name):
        seen += 1

        if not markup_path.is_file():
            skipped_no_markup += 1
            _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
            continue

        markup = read_markup(markup_path)
        if markup is None:
            skipped_no_markup += 1
            _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
            continue

        module_raw = extract_module_size_px(markup)
        expected = extract_expected_value(markup)
        if module_raw is None:
            skipped_bad_module += 1
            _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
            continue
        if expected is None:
            expected = ""

        module_binned = quantize_module_size(module_raw, cfg.module_bin_step_px)
        if module_binned is None:
            skipped_bad_module += 1
            _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
            continue

        decoded_values: list[str] = []
        times: list[float] = []
        decode_failed = False

        for _ in range(iterations):
            try:
                decoded, time_total = engine.decode_once(image_path)
            except EngineError:
                skipped_image_read += 1
                decode_failed = True
                break

            decoded_values.append(_safe_str(decoded))
            times.append(float(time_total))

        if decode_failed or not times:
            _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
            continue

        time_min = min(times)
        acc_list = [1.0 if val.strip() else 0.0 for val in decoded_values]
        accuracy = sum(acc_list) / float(len(acc_list)) if acc_list else 0.0
        decoded_best = Counter(decoded_values).most_common(1)[0][0] if decoded_values else ""

        image_rel = _relative_path(image_path, project_root)

        results.append(
            SampleResult(
                dataset=cfg.dataset_name,
                engine=engine.name,
                module_size=int(module_binned),
                module_size_raw=float(module_raw),
                module_bin_step_px=int(cfg.module_bin_step_px),
                time_total_min_sec=float(time_min),
                time_mode=cfg.time_mode,
                accuracy=float(accuracy),
                decoded=decoded_best,
                expected=expected,
                image_path=image_rel,
                decode_iterations=int(iterations),
            )
        )
        processed += 1
        _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)

    summary = ExperimentSummary(
        processed=processed,
        skipped_no_markup=skipped_no_markup,
        skipped_bad_module=skipped_bad_module,
        skipped_image_read=skipped_image_read,
    )

    return results, summary


def save_results_json(
    project_root: Path,
    engine_name: str,
    dataset_name: str,
    results: list[SampleResult],
) -> Path:
    out_dir = project_root / "outputs" / f"{engine_name}_json_and_graphics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"qr_experiment_data_{dataset_name}_{engine_name}.json"
    payload = [asdict(item) for item in results]

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_json


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _notify_progress(
    progress_cb: Callable[[ProgressState], None] | None,
    seen: int,
    processed: int,
    skipped_no_markup: int,
    skipped_bad_module: int,
    skipped_image_read: int,
) -> None:
    if not progress_cb:
        return
    progress_cb(
        ProgressState(
            seen=seen,
            processed=processed,
            skipped_no_markup=skipped_no_markup,
            skipped_bad_module=skipped_bad_module,
            skipped_image_read=skipped_image_read,
        )
    )
