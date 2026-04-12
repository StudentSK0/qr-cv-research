from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

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
    decode_success_rate: float
    gt_accuracy: float | None
    metric_kind: str
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


@dataclass(frozen=True)
class NormalizedSampleResult:
    dataset: str
    engine: str
    image_path: str
    x_target: float
    module_size_px_current: float
    scale_factor: float
    resized_width: int
    resized_height: int
    decode_time_ms: float
    decode_success: bool
    decode_success_rate: float
    gt_accuracy: float | None
    metric_kind: str
    accuracy: float
    decoded: str
    expected: str
    decode_iterations: int
    time_mode: str


@dataclass(frozen=True)
class SweepTargetSummary:
    x_target: float
    processed: int
    skipped_no_markup: int
    skipped_bad_module: int
    skipped_image_read: int


@dataclass(frozen=True)
class NormalizationSweepSummary:
    targets: list[SweepTargetSummary]


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
        decode_success_rate, gt_accuracy, accuracy, metric_kind = _resolve_accuracy(decoded_values, expected)
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
                decode_success_rate=float(decode_success_rate),
                gt_accuracy=None if gt_accuracy is None else float(gt_accuracy),
                metric_kind=metric_kind,
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


def run_module_size_normalization_sweep(
    project_root: Path,
    engine: BaseEngine,
    cfg: ExperimentConfig,
    x_targets: Iterable[float],
    progress_cb: Callable[[ProgressState], None] | None = None,
) -> tuple[list[NormalizedSampleResult], NormalizationSweepSummary]:
    import cv2

    targets = _normalize_targets(x_targets)
    if not targets:
        raise ValueError("x_targets must contain at least one positive value.")

    if engine.__class__._decode_array is BaseEngine._decode_array:
        raise EngineError(f"Engine '{engine.name}' does not support normalization sweep.")

    iterations = max(1, int(cfg.decode_iterations))

    all_results: list[NormalizedSampleResult] = []
    summary_rows: list[SweepTargetSummary] = []

    for x_target in targets:
        processed = 0
        skipped_no_markup = 0
        skipped_bad_module = 0
        skipped_image_read = 0
        seen = 0

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
            if expected is None:
                expected = ""

            if module_raw is None or module_raw <= 0:
                skipped_bad_module += 1
                _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
                continue

            scale_factor = float(x_target) / float(module_raw)
            if not math.isfinite(scale_factor) or scale_factor <= 0:
                skipped_bad_module += 1
                _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                skipped_image_read += 1
                _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
                continue

            resized_img, resized_w, resized_h = _resize_for_target(cv2, image, scale_factor)

            decoded_values: list[str] = []
            times: list[float] = []
            decode_failed = False

            for _ in range(iterations):
                try:
                    decoded, decode_time_sec = engine.decode_array_once(resized_img)
                except EngineError:
                    skipped_image_read += 1
                    decode_failed = True
                    break

                decoded_values.append(_safe_str(decoded))
                times.append(float(decode_time_sec))

            if decode_failed or not times:
                _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)
                continue

            time_min_sec = min(times)
            decode_success_rate, gt_accuracy, accuracy, metric_kind = _resolve_accuracy(decoded_values, expected)
            decoded_best = Counter(decoded_values).most_common(1)[0][0] if decoded_values else ""
            decode_success = any(val.strip() for val in decoded_values)

            all_results.append(
                NormalizedSampleResult(
                    dataset=cfg.dataset_name,
                    engine=engine.name,
                    image_path=_relative_path(image_path, project_root),
                    x_target=float(x_target),
                    module_size_px_current=float(module_raw),
                    scale_factor=float(scale_factor),
                    resized_width=int(resized_w),
                    resized_height=int(resized_h),
                    decode_time_ms=float(time_min_sec * 1000.0),
                    decode_success=bool(decode_success),
                    decode_success_rate=float(decode_success_rate),
                    gt_accuracy=None if gt_accuracy is None else float(gt_accuracy),
                    metric_kind=metric_kind,
                    accuracy=float(accuracy),
                    decoded=decoded_best,
                    expected=expected,
                    decode_iterations=int(iterations),
                    time_mode=cfg.time_mode,
                )
            )

            processed += 1
            _notify_progress(progress_cb, seen, processed, skipped_no_markup, skipped_bad_module, skipped_image_read)

        summary_rows.append(
            SweepTargetSummary(
                x_target=float(x_target),
                processed=processed,
                skipped_no_markup=skipped_no_markup,
                skipped_bad_module=skipped_bad_module,
                skipped_image_read=skipped_image_read,
            )
        )

    return all_results, NormalizationSweepSummary(targets=summary_rows)


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


def save_normalization_sweep_json(
    project_root: Path,
    engine_name: str,
    dataset_name: str,
    x_targets: Iterable[float],
    results: list[NormalizedSampleResult],
    summary: NormalizationSweepSummary,
) -> Path:
    out_dir = project_root / "outputs" / f"{engine_name}_json_and_graphics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"qr_normalization_sweep_{dataset_name}_{engine_name}.json"
    payload = {
        "mode": "module_size_normalization_sweep",
        "dataset": dataset_name,
        "engine": engine_name,
        "x_targets": [float(x) for x in _normalize_targets(x_targets)],
        "time_mode": "time_total_min",
        "results": [asdict(item) for item in results],
        "summary": {"targets": [asdict(row) for row in summary.targets]},
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_json


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _resolve_accuracy(decoded_values: list[str], expected: str) -> tuple[float, float | None, float, str]:
    success_list = [1.0 if val.strip() else 0.0 for val in decoded_values]
    decode_success_rate = sum(success_list) / float(len(success_list)) if success_list else 0.0

    if expected.strip():
        gt_hits = [1.0 if val == expected else 0.0 for val in decoded_values]
        gt_accuracy = sum(gt_hits) / float(len(gt_hits)) if gt_hits else 0.0
        return float(decode_success_rate), float(gt_accuracy), float(gt_accuracy), "gt_accuracy"

    return float(decode_success_rate), None, float(decode_success_rate), "decode_success_rate"


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resize_for_target(cv2: Any, image: Any, scale_factor: float) -> tuple[Any, int, int]:
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * scale_factor)))
    new_h = max(1, int(round(height * scale_factor)))

    if new_w == width and new_h == height:
        return image, width, height

    interpolation = cv2.INTER_LINEAR if scale_factor > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, new_w, new_h


def _normalize_targets(x_targets: Iterable[float]) -> list[float]:
    normalized: list[float] = []
    for item in x_targets:
        value = float(item)
        if not math.isfinite(value) or value <= 0:
            continue
        normalized.append(value)

    # deterministic order + dedup while preserving sorted sweep semantics
    return sorted(set(normalized))


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
