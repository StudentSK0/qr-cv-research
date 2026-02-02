from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

from ..binning import BinStats
from ..metrics import SampleResult


def _infer_tick_step(values: Sequence[int]) -> int:
    unique = sorted({int(v) for v in values if v is not None})
    if len(unique) < 2:
        return 1
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b - a > 0]
    return min(diffs) if diffs else 1


def _smooth_series(values: Sequence[float]) -> list[float]:
    if len(values) < 3:
        return list(values)
    sigma = max(1.0, len(values) / 30.0)
    return gaussian_filter1d(np.array(values, dtype=float), sigma=sigma, mode="nearest").tolist()


def _quantize_raw_to_bin(raw_value: float, step_px: int) -> int:
    step = max(1, int(step_px))
    rounded = int(round(raw_value))
    if step == 1:
        return max(1, rounded)
    return int(round(rounded / step) * step)


def _warn_if_time_is_binned(x_time_raw: Sequence[float], step_px: int) -> None:
    if step_px <= 1 or len(x_time_raw) < 2:
        return
    unique_raw = {round(v, 6) for v in x_time_raw}
    if len(unique_raw) < 2:
        return
    binned = [_quantize_raw_to_bin(v, step_px) for v in x_time_raw]
    raw_rounded = [int(round(v)) for v in x_time_raw]
    if all(r == b for r, b in zip(raw_rounded, binned)):
        print(
            "Warning: time plot appears to use binned module sizes. "
            "Ensure raw module_size_raw_px is used for time scatter/trend.",
        )


def build_interactive_plot(
    results: Sequence[SampleResult],
    bin_stats: Sequence[BinStats],
    output_html: Path,
    title: str | None = None,
) -> None:
    if not results:
        raise ValueError("No results provided for plotting.")

    results_sorted = sorted(results, key=lambda r: r.module_size_raw)
    bins_sorted = sorted(bin_stats, key=lambda b: b.module_size)

    x_time_raw = [float(r.module_size_raw) for r in results_sorted]
    y_time_raw = [r.time_total_min_sec for r in results_sorted]

    x_acc_axis = [b.module_size for b in bins_sorted]
    tick_step = _infer_tick_step(x_acc_axis)
    y_count_acc1 = [b.count_acc1 for b in bins_sorted]
    y_count_acc0 = [b.count_acc0 for b in bins_sorted]
    bin_step_px = int(results_sorted[0].module_bin_step_px) if results_sorted else 1
    _warn_if_time_is_binned(x_time_raw, bin_step_px)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.25, 0.2],
        specs=[[{}], [{}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=x_time_raw,
            y=y_time_raw,
            mode="markers",
            name="Time total min",
            marker=dict(size=6, color="#1f77b4", opacity=0.6),
            hovertemplate="module_size_raw=%{x:.3f}<br>time=%{y:.6f}s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    y_time_smooth = _smooth_series(y_time_raw)
    fig.add_trace(
        go.Scatter(
            x=x_time_raw,
            y=y_time_smooth,
            mode="lines",
            name="Time trend",
            line=dict(color="#1f77b4", width=2, shape="spline"),
            hovertemplate="module_size_raw=%{x:.3f}<br>smoothed_time=%{y:.6f}s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    x_acc_bin = x_acc_axis
    fig.add_trace(
        go.Bar(
            x=x_acc_bin,
            y=y_count_acc1,
            name="Correct decodes",
            marker=dict(color="#2ca02c"),
            hovertemplate="module_size=%{x}<br>count_acc1=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=x_acc_bin,
            y=y_count_acc0,
            name="Failed decodes",
            marker=dict(color="#d62728"),
            hovertemplate="module_size=%{x}<br>count_acc0=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Module size px (binned)",
                    "Images (count)",
                    "Mean accuracy (%)",
                    "Correct decodes (count)",
                    "Failed decodes (count)",
                ],
                fill_color="#f0f0f0",
                align="left",
            ),
            cells=dict(
                values=[
                    x_acc_bin,
                    [b.count for b in bins_sorted],
                    [f"{b.mean_accuracy * 100:.2f}" for b in bins_sorted],
                    y_count_acc1,
                    y_count_acc0,
                ],
                align="left",
            ),
        ),
        row=3,
        col=1,
    )

    max_time = max(y_time_raw, default=0.0)
    max_count = max(y_count_acc1 + y_count_acc0, default=0)
    max_x = max(max(x_time_raw, default=0.0), max(x_acc_axis, default=0)) if (x_time_raw or x_acc_axis) else 1

    fig.update_layout(
        title=title or "QR decoding vs module size",
        barmode="stack",
        hovermode="closest",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    x_label = "Module size (px)"
    fig.update_xaxes(
        title_text=x_label,
        type="linear",
        tickmode="linear",
        tick0=0,
        dtick=tick_step,
        showticklabels=True,
        ticks="outside",
        range=[0, max_x],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text=x_label,
        type="linear",
        tickmode="linear",
        tick0=0,
        dtick=tick_step,
        showticklabels=True,
        ticks="outside",
        range=[0, max_x],
        row=1,
        col=1,
    )

    fig.update_yaxes(title_text="Time total min (sec)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    time_upper = max_time * 1.05 if max_time > 0 else 1.0
    fig.update_yaxes(range=[0, time_upper], row=1, col=1)

    count_upper = max_count * 1.05 if max_count > 0 else 1.0
    fig.update_yaxes(range=[0, count_upper], row=2, col=1)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs=True, full_html=True)
