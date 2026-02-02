from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import SampleResult


@dataclass(frozen=True)
class BinStats:
    module_size: int
    count: int
    mean_accuracy: float
    count_acc1: int
    count_acc0: int
    mean_time: float


def quantize_module_size(module_size_px: float | None, step_px: int) -> int | None:
    if module_size_px is None:
        return None

    step = max(1, int(step_px))
    rounded = int(round(module_size_px))
    if step == 1:
        return max(1, rounded)

    quantized = int(round(rounded / step) * step)
    return max(step, quantized)


def build_bin_stats(results: Iterable[SampleResult]) -> list[BinStats]:
    bins: dict[int, list[SampleResult]] = defaultdict(list)
    for record in results:
        bins[record.module_size].append(record)

    stats: list[BinStats] = []
    for module_size in sorted(bins.keys()):
        items = bins[module_size]
        count = len(items)
        if count == 0:
            continue

        mean_accuracy = sum(r.accuracy for r in items) / float(count)
        mean_time = sum(r.time_total_min_sec for r in items) / float(count)
        count_acc1 = sum(1 for r in items if r.accuracy == 1.0)
        count_acc0 = sum(1 for r in items if r.accuracy == 0.0)

        stats.append(
            BinStats(
                module_size=module_size,
                count=count,
                mean_accuracy=mean_accuracy,
                count_acc1=count_acc1,
                count_acc0=count_acc0,
                mean_time=mean_time,
            )
        )

    return stats
