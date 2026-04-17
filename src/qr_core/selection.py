from __future__ import annotations

import math
from typing import Any, Sequence

from .pareto import pareto_front


def select_optimal_sweep_target(
    target_stats: Sequence[dict[str, Any]],
    *,
    eps_pp: float = 2.0,
    lambda_spread: float = 0.15,
    use_pareto_prefilter: bool = True,
    quality_key: str | None = None,
) -> dict[str, Any]:
    """Select recommended x_target for sweep using plateau-first logic.

    The algorithm prioritizes quality and then minimizes time:
    1) Build quality plateau: quality >= max_quality - eps_pp.
    2) Optionally apply Pareto prefilter on (min time_median_ms, min spread_ms, max quality).
    3) Inside candidates, minimize score = time_median_ms + lambda_spread * spread_ms.
    """
    points = _normalize_target_points(target_stats)
    if not points:
        return {
            "quality_metric": "success_rate",
            "quality_max": 0.0,
            "plateau_threshold": 0.0,
            "eps_pp": float(max(0.0, eps_pp)),
            "lambda_spread": float(max(0.0, lambda_spread)),
            "pareto_points": [],
            "plateau_points": [],
            "candidate_points": [],
            "recommended": None,
        }

    quality_metric = _resolve_quality_key(target_stats, quality_key)
    eps_pp = float(max(0.0, eps_pp))
    lambda_spread = float(max(0.0, lambda_spread))

    quality_values = [float(p[quality_metric]) for p in points]
    quality_max = max(quality_values)
    plateau_threshold = quality_max - eps_pp

    plateau_points = [p for p in points if float(p[quality_metric]) >= plateau_threshold]

    pareto_input = [
        {
            "module_size": float(p["x_target"]),  # deterministic order key in pareto_front
            "x_target": float(p["x_target"]),
            "time_median_ms": float(p["time_median_ms"]),
            "spread_ms": float(p["spread_ms"]),
            "quality_rate": float(p[quality_metric]),
        }
        for p in points
    ]
    pareto_points_raw = pareto_front(
        pareto_input,
        minimize_keys=("time_median_ms", "spread_ms"),
        maximize_keys=("quality_rate",),
    )
    pareto_x = {float(p["x_target"]) for p in pareto_points_raw}
    pareto_points = [p for p in points if float(p["x_target"]) in pareto_x]

    if use_pareto_prefilter:
        candidate_points = [p for p in plateau_points if float(p["x_target"]) in pareto_x]
        if not candidate_points:
            candidate_points = plateau_points
    else:
        candidate_points = plateau_points

    scored_candidates: list[dict[str, float]] = []
    for p in candidate_points:
        score = float(p["time_median_ms"]) + lambda_spread * float(p["spread_ms"])
        scored = dict(p)
        scored["score"] = score
        scored_candidates.append(scored)

    recommended = min(
        scored_candidates,
        key=lambda p: (
            float(p["score"]),
            float(p["time_median_ms"]),
            -float(p[quality_metric]),
            float(p["x_target"]),
        ),
    )

    return {
        "quality_metric": quality_metric,
        "quality_max": float(quality_max),
        "plateau_threshold": float(plateau_threshold),
        "eps_pp": eps_pp,
        "lambda_spread": lambda_spread,
        "pareto_points": sorted(pareto_points, key=lambda p: float(p["x_target"])),
        "plateau_points": sorted(plateau_points, key=lambda p: float(p["x_target"])),
        "candidate_points": sorted(scored_candidates, key=lambda p: float(p["x_target"])),
        "recommended": recommended,
    }


def _resolve_quality_key(target_stats: Sequence[dict[str, Any]], preferred: str | None) -> str:
    if preferred in {"accuracy_rate", "success_rate"}:
        return str(preferred)

    has_gt = any(_as_float(row.get("gt_count")) > 0.0 for row in target_stats)
    return "accuracy_rate" if has_gt else "success_rate"


def _normalize_target_points(target_stats: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in target_stats:
        x_target = _as_float(row.get("x_target"))
        time_median = _as_float(row.get("time_median_ms"))
        p10 = _as_float(row.get("time_p10_ms"))
        p90 = _as_float(row.get("time_p90_ms"))
        success_rate = _as_float(row.get("success_rate"))
        accuracy_rate = _as_float(row.get("accuracy_rate"))
        if not (math.isfinite(x_target) and math.isfinite(time_median)):
            continue
        if not math.isfinite(p10):
            p10 = 0.0
        if not math.isfinite(p90):
            p90 = p10
        spread = max(0.0, p90 - p10)
        if not math.isfinite(success_rate):
            success_rate = 0.0
        if not math.isfinite(accuracy_rate):
            accuracy_rate = success_rate

        points.append(
            {
                "x_target": float(x_target),
                "time_median_ms": float(time_median),
                "spread_ms": float(spread),
                "success_rate": float(success_rate),
                "accuracy_rate": float(accuracy_rate),
            }
        )
    return points


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result

