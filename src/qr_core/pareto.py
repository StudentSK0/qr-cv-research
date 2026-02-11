from __future__ import annotations

from typing import Iterable, Sequence, Any

import numpy as np


def pareto_front(
    points: Sequence[dict[str, float]],
    minimize_keys: Sequence[str],
    maximize_keys: Sequence[str],
) -> list[dict[str, float]]:
    """Return Pareto-optimal points with deterministic ordering.

    A dominates B if A is <= on all minimize_keys and >= on all maximize_keys,
    and at least one inequality is strict.
    """
    if not points:
        return []

    def sort_key(p: dict[str, float]) -> tuple:
        minimize = tuple(float(p[k]) for k in minimize_keys)
        maximize = tuple(-float(p[k]) for k in maximize_keys)
        module_size = (float(p.get("module_size", 0.0)),)
        return minimize + maximize + module_size

    ordered = sorted(points, key=sort_key)
    front: list[dict[str, float]] = []

    for p in ordered:
        dominated = False
        for q in ordered:
            if q is p:
                continue
            if _dominates(q, p, minimize_keys, maximize_keys):
                dominated = True
                break
        if not dominated:
            front.append(p)

    return front


def aggregate_module_sizes(
    results: Iterable[Any],
    n_min: int = 10,
    time_attr_candidates: Sequence[str] = ("time_total_min_sec", "time_total_min", "time"),
    accuracy_attr: str = "accuracy",
    module_size_attr: str = "module_size",
) -> list[dict[str, float]]:
    """Aggregate metrics per module size (binned).

    Returns list of dicts with:
    module_size, n, time_median, time_p90, accuracy_mean
    """
    groups: dict[int, list[Any]] = {}
    for item in results:
        module_size = _get_value(item, module_size_attr)
        if module_size is None:
            continue
        module_size_int = int(round(float(module_size)))
        groups.setdefault(module_size_int, []).append(item)

    stats: list[dict[str, float]] = []
    for module_size, items in sorted(groups.items()):
        times = [_get_time_value(x, time_attr_candidates) for x in items]
        times = [t for t in times if t is not None]
        if not times:
            continue
        accuracies = [_get_value(x, accuracy_attr) for x in items]
        accuracies = [a for a in accuracies if a is not None]

        n = len(times)
        if n < n_min:
            continue

        time_median = float(np.median(times))
        time_p90 = float(np.percentile(times, 90))
        accuracy_mean = float(np.mean(accuracies)) if accuracies else 0.0

        stats.append(
            {
                "module_size": float(module_size),
                "n": float(n),
                "time_median": time_median,
                "time_p90": time_p90,
                "accuracy_mean": accuracy_mean,
            }
        )

    return stats


def _dominates(
    a: dict[str, float],
    b: dict[str, float],
    minimize_keys: Sequence[str],
    maximize_keys: Sequence[str],
) -> bool:
    all_min = all(float(a[k]) <= float(b[k]) for k in minimize_keys)
    all_max = all(float(a[k]) >= float(b[k]) for k in maximize_keys)
    if not (all_min and all_max):
        return False

    any_strict = any(float(a[k]) < float(b[k]) for k in minimize_keys) or any(
        float(a[k]) > float(b[k]) for k in maximize_keys
    )
    return any_strict


def _get_value(item: Any, key: str) -> float | None:
    if isinstance(item, dict):
        value = item.get(key)
    else:
        value = getattr(item, key, None)
    if value is None:
        return None
    return float(value)


def _get_time_value(item: Any, keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _get_value(item, key)
        if value is not None:
            return float(value)
    return None


if __name__ == "__main__":
    sample = [
        {"module_size": 2, "time": 0.2, "accuracy": 1.0},
        {"module_size": 2, "time": 0.25, "accuracy": 1.0},
        {"module_size": 4, "time": 0.15, "accuracy": 0.5},
        {"module_size": 6, "time": 0.12, "accuracy": 0.6},
        {"module_size": 8, "time": 0.3, "accuracy": 0.9},
    ]
    stats = aggregate_module_sizes(sample, n_min=1, time_attr_candidates=("time",))
    front = pareto_front(stats, minimize_keys=("time_median",), maximize_keys=("accuracy_mean",))
    print("Pareto front:")
    for item in front:
        print(item)
