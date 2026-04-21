from __future__ import annotations

import json
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..metrics import NormalizationSweepSummary, NormalizedSampleResult
from ..selection import select_optimal_sweep_target


def _is_no_gt(expected: str | None) -> bool:
    return not str(expected or "").strip()


def _truncate(text: str, max_len: int = 88) -> str:
    value = str(text or "")
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    q = max(0.0, min(100.0, float(q)))
    idx = (len(ordered) - 1) * (q / 100.0)
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _build_target_stats(results: Sequence[NormalizedSampleResult]) -> list[dict[str, Any]]:
    grouped: dict[float, list[NormalizedSampleResult]] = {}
    for item in results:
        grouped.setdefault(float(item.x_target), []).append(item)

    stats: list[dict[str, Any]] = []
    for x_target in sorted(grouped):
        rows = grouped[x_target]
        times = [float(r.decode_time_ms) for r in rows]
        accuracies = [float(r.accuracy) for r in rows]
        no_gt_count = sum(1 for r in rows if _is_no_gt(r.expected))
        gt_count = len(rows) - no_gt_count
        time_p10 = _percentile(times, 10.0)
        time_p90 = _percentile(times, 90.0)
        accuracy_rate = float(mean(accuracies) * 100.0) if accuracies else 0.0

        stats.append(
            {
                "x_target": float(x_target),
                "count": len(rows),
                "gt_count": gt_count,
                "no_gt_count": no_gt_count,
                # keep alias for backward compatibility in helper code paths;
                # reporting/selection use accuracy_rate as the single metric.
                "success_rate": float(accuracy_rate),
                "accuracy_rate": float(accuracy_rate),
                "time_mean_ms": float(mean(times)) if times else 0.0,
                "time_median_ms": float(median(times)) if times else 0.0,
                "time_p10_ms": time_p10,
                "time_p90_ms": time_p90,
                "spread_ms": max(0.0, float(time_p90) - float(time_p10)),
            }
        )

    return stats


def _build_plot_html(target_stats: Sequence[dict[str, Any]], title: str) -> str:
    x = [row["x_target"] for row in target_stats]
    y_time_median = [row["time_median_ms"] for row in target_stats]
    y_time_mean = [row["time_mean_ms"] for row in target_stats]
    y_time_p10 = [row["time_p10_ms"] for row in target_stats]
    y_time_p90 = [row["time_p90_ms"] for row in target_stats]
    y_accuracy = [row["accuracy_rate"] for row in target_stats]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        row_heights=[0.60, 0.40],

    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_time_p90,
            mode="lines",
            name="p90",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_time_p10,
            mode="lines",
            name="p10-p90 band",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(14, 165, 233, 0.14)",
            hovertemplate="module size=%{x:.0f}px<br>p10=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_time_median,
            mode="lines+markers",
            name="Median time",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=8),
            hovertemplate="module size=%{x:.0f}px<br>median=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_time_mean,
            mode="lines+markers",
            name="Mean time",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=7),
            hovertemplate="module size=%{x:.0f}px<br>mean=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_accuracy,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=8),
            hovertemplate="module size=%{x:.0f}px<br>accuracy=%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )


    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.99, yanchor="top"),
        height=820,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0,
            bgcolor="rgba(255, 255, 255, 0.92)",
        ),
        margin=dict(l=54, r=24, t=150, b=56),
    )
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", range=[-10, 110], row=2, col=1)

    x_axis_cfg: dict[str, Any]
    unique_x = sorted(set(float(v) for v in x))
    all_integer_x = bool(unique_x) and all(abs(v - round(v)) < 1e-9 for v in unique_x)
    if all_integer_x:
        min_x = int(round(unique_x[0]))
        max_x = int(round(unique_x[-1]))
        x_axis_cfg = {
            "type": "linear",
            "tickmode": "linear",
            "tick0": min_x,
            "dtick": 1,
            "range": [min_x - 0.5, max_x + 0.5],
        }
    else:
        x_axis_cfg = {
            "tickmode": "array",
            "tickvals": unique_x,
            "ticktext": [f"{v:g}" for v in unique_x],
        }

    fig.update_xaxes(title_text="Module size (px)", showticklabels=True, row=1, col=1, **x_axis_cfg)
    fig.update_xaxes(title_text="Module size (px)", row=2, col=1, **x_axis_cfg)

    return fig.to_html(full_html=False, include_plotlyjs=True)


def build_sweep_report(
    results: Sequence[NormalizedSampleResult],
    summary: NormalizationSweepSummary,
    output_html: Path,
    *,
    dataset: str,
    engine: str,
    iterations: int,
    time_mode: str,
    x_targets: Sequence[float],
    title: str | None = None,
) -> None:
    if not results:
        raise ValueError("No sweep results provided for plotting.")

    sorted_results = sorted(results, key=lambda r: (float(r.x_target), str(r.image_path)))
    target_stats = _build_target_stats(sorted_results)

    total_samples = len(sorted_results)
    no_gt_samples = sum(1 for item in sorted_results if _is_no_gt(item.expected))
    gt_samples = total_samples - no_gt_samples
    total_skipped_no_markup = sum(
        int(getattr(row, "skipped_no_markup", 0)) for row in summary.targets
    )

    summary_by_target: dict[float, Any] = {
        float(row.x_target): row for row in summary.targets
    }

    x_targets_text = ", ".join(f"{float(x):g}" for x in sorted(set(float(x) for x in x_targets)))
    optimal_analysis = select_optimal_sweep_target(
        target_stats,
        eps_pp=2.0,
        lambda_spread=0.15,
        use_pareto_prefilter=True,
    )
    recommended = optimal_analysis.get("recommended")
    def _format_x_targets(items: Sequence[dict[str, Any]]) -> str:
        if not items:
            return "n/a"
        return ", ".join(f"{float(item['x_target']):g}" for item in items)

    plateau_targets_text = _format_x_targets(optimal_analysis.get("plateau_points") or [])
    candidate_targets_text = _format_x_targets(optimal_analysis.get("candidate_points") or [])
    pareto_targets_text = _format_x_targets(optimal_analysis.get("pareto_points") or [])


    if isinstance(recommended, dict):
        recommended_x_target_text = f"{float(recommended['x_target']):g} px"
        recommended_quality_text = f"{float(recommended['accuracy_rate']):.2f}%"
        recommended_time_text = f"{float(recommended['time_median_ms']):.3f} ms"
        recommended_spread_text = f"{float(recommended['spread_ms']):.3f} ms"
    else:
        recommended_x_target_text = "n/a"
        recommended_quality_text = "n/a"
        recommended_time_text = "n/a"
        recommended_spread_text = "n/a"

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        return parsed

    max_quality_point: dict[str, Any] | None = None
    if target_stats:
        max_quality_point = max(
            target_stats,
            key=lambda row: (
                _safe_float(row.get("accuracy_rate"), 0.0),
                -_safe_float(row.get("time_median_ms"), float("inf")),
                -_safe_float(row.get("spread_ms"), float("inf")),
                -_safe_float(row.get("x_target"), float("inf")),
            ),
        )

    if max_quality_point is not None:
        max_quality_x_target_text = f"{_safe_float(max_quality_point.get('x_target')):g} px"
        max_quality_quality_text = f"{_safe_float(max_quality_point.get('accuracy_rate')):.2f}%"
        max_quality_time_text = f"{_safe_float(max_quality_point.get('time_median_ms')):.3f} ms"
        max_quality_spread_text = f"{_safe_float(max_quality_point.get('spread_ms')):.3f} ms"
    else:
        max_quality_x_target_text = "n/a"
        max_quality_quality_text = "n/a"
        max_quality_time_text = "n/a"
        max_quality_spread_text = "n/a"

    target_summary_rows = []
    for row in target_stats:
        x_target = float(row["x_target"])
        target_info = summary_by_target.get(x_target)

        target_summary_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{x_target:g}</td>",
                    f"<td>{int(row['count'])}</td>",
                    f"<td>{row['accuracy_rate']:.2f}%</td>",
                    f"<td>{row['time_median_ms']:.3f}</td>",
                    f"<td>{row['time_mean_ms']:.3f}</td>",
                    f"<td>{row['time_p90_ms']:.3f}</td>",
                    f"<td>{row['time_p10_ms']:.3f}</td>",
                    f"<td>{int(row['gt_count'])}</td>",
                    "</tr>",
                ]
            )
        )

    sample_data: list[dict[str, Any]] = []
    table_rows: list[str] = []

    for idx, item in enumerate(sorted_results):
        no_gt = _is_no_gt(item.expected)
        metric_kind_raw = str(getattr(item, "metric_kind", "") or "").strip()
        if metric_kind_raw in {"gt_accuracy", "decode_success_rate"}:
            metric_kind = metric_kind_raw
        else:
            gt_accuracy_probe = getattr(item, "gt_accuracy", None)
            if (not no_gt) and gt_accuracy_probe is not None:
                metric_kind = "gt_accuracy"
            else:
                metric_kind = "decode_success_rate"
        decode_success_rate = float(getattr(item, "decode_success_rate", 1.0 if item.decode_success else 0.0))
        gt_accuracy_raw = getattr(item, "gt_accuracy", None)
        gt_accuracy_value = float(gt_accuracy_raw) if gt_accuracy_raw is not None else None
        accuracy_value: float | None = float(item.accuracy) if item.accuracy is not None else (
            decode_success_rate if metric_kind == "decode_success_rate" else gt_accuracy_value
        )

        decoded_text = str(item.decoded or "")
        expected_text = str(item.expected or "")

        sample_payload = {
            "sample_index": idx,
            "dataset": dataset,
            "image_path": str(item.image_path),
            "x_target": float(item.x_target),
            "module_size_px_current": float(item.module_size_px_current),
            "scale_factor": float(item.scale_factor),
            "resized_width": int(item.resized_width),
            "resized_height": int(item.resized_height),
            "decode_time_ms": float(item.decode_time_ms),
            "decode_success": bool(item.decode_success),
            "decoded": decoded_text,
            "expected": expected_text,
            "metric_kind": metric_kind,
            "decode_success_rate": decode_success_rate,
            "gt_accuracy": gt_accuracy_value,
            "accuracy": accuracy_value,
            "no_gt": no_gt,
        }
        sample_data.append(sample_payload)

        expected_display = "N/A (no ground-truth)" if no_gt else _truncate(expected_text)
        accuracy_display = "n/a" if accuracy_value is None else f"{float(accuracy_value):.3f}"
        no_gt_badge = '<span class="no-gt-badge">NO GT</span>' if no_gt else ""

        table_rows.append(
            "".join(
                [
                    f'<tr data-index="{idx}">',
                    f'<td data-value="{idx}">{idx + 1}</td>',
                    f'<td data-value="{escape(str(item.image_path))}">{escape(str(item.image_path))}</td>',
                    f'<td data-value="{float(item.x_target):.6f}">{float(item.x_target):g}</td>',
                    f'<td data-value="{float(item.module_size_px_current):.6f}">{float(item.module_size_px_current):.3f}</td>',
                    f'<td data-value="{float(item.scale_factor):.8f}">{float(item.scale_factor):.3f}</td>',
                    f'<td data-value="{int(item.resized_width) * int(item.resized_height)}">{int(item.resized_width)} x {int(item.resized_height)}</td>',
                    f'<td data-value="{float(item.decode_time_ms):.8f}">{float(item.decode_time_ms):.3f}</td>',
                    f'<td data-value="{accuracy_value if accuracy_value is not None else -1}">{accuracy_display}</td>',
                    f'<td data-value="{escape(decoded_text)}">{escape(_truncate(decoded_text) or "N/A")}</td>',
                    f'<td data-value="{escape(expected_text)}">{escape(expected_display)} {no_gt_badge}</td>',
                    f'<td data-value="{idx}"><button class="details-btn" type="button" data-index="{idx}">Details</button></td>',
                    "</tr>",
                ]
            )
        )

    report_title = title or f"Normalization sweep ({dataset}, {engine})"
    plot_html = _build_plot_html(target_stats, report_title)

    html_parts: list[str] = []
    html_parts.append(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Normalization sweep report</title>
    <style>
      :root {
        --bg: #eef3f8;
        --card: #ffffff;
        --line: #dbe3ef;
        --ink: #0f172a;
        --muted: #475569;
        --accent: #0f8b8d;
        --accent-2: #1d4ed8;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        padding: 20px;
        font-family: "Space Grotesk", "Sora", "IBM Plex Sans", sans-serif;
        background: var(--bg);
        color: var(--ink);
      }
      .layout {
        max-width: 1400px;
        margin: 0 auto;
        display: grid;
        gap: 16px;
      }
      .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
      }
      .summary-item {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 12px;
        background: #f8fbff;
      }
      .summary-item__label {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 4px;
      }
      .summary-item__value {
        font-size: 15px;
        font-weight: 600;
      }
      .summary-help {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        margin-left: 4px;
        border-radius: 50%;
        background: #e2e8f0;
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        cursor: help;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        table-layout: fixed;
      }
      th, td {
        border: 1px solid var(--line);
        padding: 8px;
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
        word-break: break-word;
        hyphens: auto;
      }
      th {
        background: #f5f8fc;
        cursor: pointer;
        user-select: none;
      }
      th.nosort {
        cursor: default;
      }
      .table-wrap {
        width: 100%;
        max-height: 520px;
        overflow: auto;
      }
      .toolbar {
        margin-bottom: 10px;
        display: flex;
        gap: 10px;
        align-items: center;
      }
      .toolbar input {
        max-width: 360px;
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--line);
      }
      .details-btn {
        border: none;
        background: var(--accent);
        color: #fff;
        border-radius: 8px;
        padding: 5px 9px;
        cursor: pointer;
        font-size: 12px;
      }
      .details-btn:hover {
        filter: brightness(1.05);
      }
      .no-gt-badge {
        display: inline-flex;
        align-items: center;
        margin-left: 4px;
        padding: 1px 6px;
        font-size: 11px;
        border-radius: 999px;
        background: #e2e8f0;
        color: #334155;
      }
      .modal {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        background: rgba(15, 23, 42, 0.45);
        z-index: 999;
      }
      .modal.open { display: flex; }
      .modal__content {
        width: min(860px, 94vw);
        max-height: 88vh;
        overflow: auto;
        background: #fff;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid var(--line);
      }
      .modal__header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
      }
      .modal__close {
        border: none;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        cursor: pointer;
        background: #e2e8f0;
        color: #1e293b;
        font-size: 18px;
      }
      .modal__body {
        margin-top: 10px;
        color: #334155;
        font-size: 14px;
        line-height: 1.45;
      }
      .modal__meta {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 8px;
        margin-bottom: 10px;
      }
      .modal__meta div {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px;
        background: #f8fbff;
      }
      .modal__note {
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        color: #334155;
        margin-top: 10px;
      }
      .modal__actions {
        margin-top: 10px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
      }
      .modal__run-result {
        font-size: 12px;
        color: #334155;
        line-height: 1.35;
        white-space: pre-line;
      }
      .sample-list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 8px;
      }
      .sample-item {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 9px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
      }
      .sample-item__text {
        display: grid;
        gap: 3px;
        min-width: 0;
      }
      .sample-item__text a {
        color: #0f766e;
        text-decoration: none;
      }
      .sample-item__text a:hover { text-decoration: underline; }
      .sample-item__meta {
        color: #64748b;
        font-size: 12px;
      }
      .sample-item__actions {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 6px;
      }
      .sample-item__button-row {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
      }
      .sample-item__run-result {
        font-size: 12px;
        color: #334155;
        text-align: right;
        line-height: 1.35;
        max-width: 320px;
      }
      .small {
        font-size: 12px;
        color: var(--muted);
      }
      .explain {
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid rgba(30, 40, 60, 0.12);
        background: #f8fbff;
        color: #334155;
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <div class="layout">
"""
    )

    html_parts.append(
        "".join(
            [
                '<section class="card">',
                f"<h1 style=\"margin:0 0 10px;\">{escape(report_title)}</h1>",
                '<div class="summary-grid">',
                f'<div class="summary-item"><div class="summary-item__label">Dataset</div><div class="summary-item__value">{escape(dataset)}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">Engine</div><div class="summary-item__value">{escape(engine)}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">Iterations</div><div class="summary-item__value">{int(iterations)}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">module sizes <span class="summary-help" title="Module sizes are target QR module sizes in pixels. The whole dataset is normalized to the first module size, then to each next size, and one combined summary plot is built.">?</span></div><div class="summary-item__value">{escape(x_targets_text)}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">Total samples</div><div class="summary-item__value">{total_samples}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">GT samples</div><div class="summary-item__value">{gt_samples}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">skip no markup</div><div class="summary-item__value">{total_skipped_no_markup}</div></div>',
                "</div>",
                '<div class="explain">P10-P90 band is the percentile range from the 10th to the 90th percentile of decode time. It shows the typical spread without extreme outliers.</div>',
                "</section>",
            ]
        )
    )

    html_parts.append(
        "".join(
            [
                '<section class="card">',
                '<h2 style="margin:0 0 10px;">Optimal module size analysis</h2>',
                '<h3 style="margin:0 0 8px;">Recommended</h3>',
                '<div class="summary-grid">',
                f'<div class="summary-item"><div class="summary-item__label">module size</div><div class="summary-item__value">{recommended_x_target_text}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">accuracy</div><div class="summary-item__value">{recommended_quality_text}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">median time</div><div class="summary-item__value">{recommended_time_text}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">time spread p90-p10</div><div class="summary-item__value">{recommended_spread_text}</div></div>',
                "</div>",
                '<h3 style="margin:12px 0 8px;">Max Accuracy</h3>',
                '<div class="summary-grid">',
                f'<div class="summary-item"><div class="summary-item__label">module size</div><div class="summary-item__value">{max_quality_x_target_text}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">accuracy</div><div class="summary-item__value">{max_quality_quality_text}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">median time</div><div class="summary-item__value">{max_quality_time_text}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">time spread p90-p10</div><div class="summary-item__value">{max_quality_spread_text}</div></div>',
                "</div>",
                "".join(
                    [
                        '<div class="explain">',
                        "Method:<br>",
                        "1) Finding the beginning of plateau at the accuracy plot by the following rule: ",
                        f"max_accuracy - &epsilon;, where &epsilon; = {float(optimal_analysis['eps_pp']):.2f} pp.<br>",
                        "2) Perfoming Pareto analysis among the values in the plateau.<br>",
                        f'Candidates: {candidate_targets_text}.',
                        "</div>",
                    ]
                ),
                "</section>",
            ]
        )
    )

    html_parts.append('<section class="card">')
    html_parts.append(plot_html)
    html_parts.append('</section>')

    html_parts.append(
        "".join(
            [
                '<section class="card">',
                '<h2 style="margin:0 0 10px;">Per-target summary</h2>',
                '<div class="table-wrap">',
                '<table id="targetSummary">',
                '<thead><tr>',
                '<th>module size</th><th>samples</th><th>accuracy</th><th>median ms</th><th>mean ms</th><th>p90 ms</th><th>p10 ms</th><th>GT</th>',
                '</tr></thead>',
                '<tbody>',
                ''.join(target_summary_rows),
                '</tbody>',
                '</table>',
                '</div>',
                '</section>',
            ]
        )
    )

    html_parts.append(
        "".join(
            [
                '<section class="card">',
                '<h2 style="margin:0 0 10px;">Per-image results</h2>',
                '<div class="toolbar">',
                '<input id="sampleSearch" type="search" placeholder="Search image path / decoded / expected...">',
                '<span class="small">Click headers to sort. Accuracy per image: with GT -> decoded == expected; without GT -> non-empty decode.</span>',
                '</div>',
                '<div class="table-wrap">',
                '<table id="samplesTable">',
                '<thead><tr>',
                '<th data-type="number">#</th>',
                '<th data-type="text">image path</th>',
                '<th data-type="number">module size px</th>',
                '<th data-type="number">initial module size px</th>',
                '<th data-type="number">scale factor</th>',
                '<th data-type="text">resized</th>',
                '<th data-type="number">time ms</th>',
                '<th data-type="number">accuracy</th>',
                '<th data-type="text">decoded</th>',
                '<th data-type="text">expected</th>',
                '<th class="nosort">details</th>',
                '</tr></thead>',
                '<tbody>',
                ''.join(table_rows),
                '</tbody>',
                '</table>',
                '</div>',
                '</section>',
            ]
        )
    )

    html_parts.append(
        """
      <div id="detailsModal" class="modal" aria-hidden="true">
        <div class="modal__content" role="dialog" aria-modal="true" aria-labelledby="detailsTitle">
          <div class="modal__header">
            <h3 id="detailsTitle" style="margin:0;">Sample details</h3>
            <button id="detailsClose" class="modal__close" type="button" aria-label="Close">x</button>
          </div>
          <div class="modal__body">
            <div id="detailsMeta" class="modal__meta"></div>
            <div><strong>decoded</strong>: <span id="detailsDecoded"></span></div>
            <div style="margin-top:6px;"><strong>expected (if provided)</strong>: <span id="detailsExpected"></span></div>
            <div id="detailsNoGt" class="modal__note" style="display:none;">Ground-truth is not provided for this image. Accuracy is reported as decode_success_rate.</div>
            <div id="detailsActions" class="modal__actions">
              <button id="detailsRunBtn" class="details-btn" type="button">Run on this image</button>
              <div id="detailsRunResult" class="modal__run-result"></div>
            </div>
            <div style="margin-top:10px;"><img id="detailsImage" alt="sample" style="max-width:100%; max-height:320px; border:1px solid var(--line); border-radius:10px; display:none;"></div>
          </div>
        </div>
      </div>

      <div id="drilldownModal" class="modal" aria-hidden="true">
        <div class="modal__content" role="dialog" aria-modal="true" aria-labelledby="drilldownTitle">
          <div class="modal__header">
            <h3 id="drilldownTitle" style="margin:0;">Target samples</h3>
            <button id="drilldownClose" class="modal__close" type="button" aria-label="Close">x</button>
          </div>
          <div id="drilldownStatus" class="small" style="margin-bottom:8px;"></div>
          <ul id="drilldownList" class="sample-list"></ul>
        </div>
      </div>
    """
    )

    html_parts.append('<script>')
    html_parts.append(f"const SWEEP_SAMPLE_DATA = {json.dumps(sample_data, ensure_ascii=False)};\n")
    html_parts.append(f"const SWEEP_DATASET = {json.dumps(dataset, ensure_ascii=False)};\n")
    html_parts.append(
        """
const queryParams = new URLSearchParams(window.location.search);
const jobId = queryParams.get("job_id");

function normalizeImagePath(pathValue) {
  if (!pathValue) return "";
  let p = String(pathValue).replace(/\\\\/g, "/").replace(/^\\/+/, "");
  const prefix = `datasets/${SWEEP_DATASET}/`;
  if (p.startsWith(prefix)) {
    p = p.slice(prefix.length);
  }
  if (p.startsWith("images/QR_CODE/")) {
    p = p.slice("images/QR_CODE/".length);
  }
  return p;
}

function byId(id) {
  return document.getElementById(id);
}

function setModalOpen(modalEl, isOpen) {
  if (!modalEl) return;
  modalEl.classList.toggle("open", isOpen);
  modalEl.setAttribute("aria-hidden", isOpen ? "false" : "true");
}

const detailsModal = byId("detailsModal");
const detailsClose = byId("detailsClose");
const detailsRunBtn = byId("detailsRunBtn");
const detailsRunResult = byId("detailsRunResult");
const detailsActions = byId("detailsActions");
const drilldownModal = byId("drilldownModal");
const drilldownClose = byId("drilldownClose");
let detailsRunContext = null;

if (detailsClose) {
  detailsClose.addEventListener("click", () => setModalOpen(detailsModal, false));
}
if (drilldownClose) {
  drilldownClose.addEventListener("click", () => setModalOpen(drilldownModal, false));
}
if (detailsModal) {
  detailsModal.addEventListener("click", (event) => {
    if (event.target === detailsModal) {
      setModalOpen(detailsModal, false);
    }
  });
}
if (drilldownModal) {
  drilldownModal.addEventListener("click", (event) => {
    if (event.target === drilldownModal) {
      setModalOpen(drilldownModal, false);
    }
  });
}

if (detailsRunBtn) {
  detailsRunBtn.addEventListener("click", () => {
    if (!detailsRunContext) {
      if (detailsRunResult) {
        detailsRunResult.textContent = "No sample selected.";
      }
      return;
    }
    runSingleSample(
      detailsRunContext.imagePath,
      detailsRunContext.xTarget,
      detailsRunBtn,
      detailsRunResult,
    );
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setModalOpen(detailsModal, false);
    setModalOpen(drilldownModal, false);
  }
});

async function fetchSampleDetails(index) {
  if (!jobId) {
    return SWEEP_SAMPLE_DATA[index] || null;
  }
  try {
    const resp = await fetch(`/api/jobs/${jobId}/sample/${index}`);
    if (!resp.ok) {
      return SWEEP_SAMPLE_DATA[index] || null;
    }
    return await resp.json();
  } catch (err) {
    return SWEEP_SAMPLE_DATA[index] || null;
  }
}

async function runSingleSample(imagePath, xTarget, button, resultEl) {
  if (!jobId) {
    resultEl.textContent = "job_id is missing in URL.";
    return;
  }
  if (!imagePath) {
    resultEl.textContent = "Missing image path.";
    return;
  }
  const targetValue = Number(xTarget);
  if (!Number.isFinite(targetValue)) {
    resultEl.textContent = "Missing x_target.";
    return;
  }

  button.disabled = true;
  resultEl.textContent = "Running...";
  try {
    const resp = await fetch(`/api/jobs/${jobId}/run_single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_path: imagePath, x_target: targetValue }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      resultEl.textContent = data && data.error ? data.error : "Failed.";
      return;
    }

    const time = data.time_total_min_sec != null ? Number(data.time_total_min_sec).toFixed(6) : "n/a";
    const acc = data.accuracy != null ? Number(data.accuracy).toFixed(3) : "n/a";
    const metricKind = String(data.metric_kind || "").trim() || "accuracy";
    const decoded = String(data.decoded || data.decoded_best || "");
    const expected = String(data.expected || "");
    resultEl.textContent = `time=${time}s, ${metricKind}=${acc}, decoded="${decoded}", expected="${expected}"`;
  } catch (err) {
    resultEl.textContent = "Request failed.";
  } finally {
    button.disabled = false;
  }
}

async function openSampleDetails(index) {
  const sample = await fetchSampleDetails(index);
  if (!sample) {
    return;
  }

  byId("detailsTitle").textContent = `Sample #${index + 1}: ${sample.image_path || ""}`;
  byId("detailsDecoded").textContent = sample.decoded || "";
  const expectedText = sample.no_gt ? "N/A (no ground-truth)" : (sample.expected || "");
  byId("detailsExpected").textContent = expectedText;

  const metricKind = String(sample.metric_kind || "").trim() || (sample.no_gt ? "decode_success_rate" : "gt_accuracy");
  const accuracyText = sample.accuracy == null ? "n/a" : Number(sample.accuracy).toFixed(3);
  const successText = sample.decode_success ? "yes" : "no";
  const meta = [
    `<div><strong>x_target</strong><br>${sample.x_target ?? ""}</div>`,
    `<div><strong>decode_time_ms</strong><br>${sample.decode_time_ms != null ? Number(sample.decode_time_ms).toFixed(3) : ""}</div>`,
    `<div><strong>decode_success</strong><br>${successText}</div>`,
    `<div><strong>${metricKind}</strong><br>${accuracyText}</div>`,
    `<div><strong>scale_factor</strong><br>${sample.scale_factor != null ? Number(sample.scale_factor).toFixed(3) : ""}</div>`,
    `<div><strong>resized</strong><br>${sample.resized_width || ""} x ${sample.resized_height || ""}</div>`,
  ];
  byId("detailsMeta").innerHTML = meta.join("");

  const noGtBox = byId("detailsNoGt");
  if (sample.no_gt) {
    noGtBox.style.display = "block";
  } else {
    noGtBox.style.display = "none";
  }

  const imgEl = byId("detailsImage");
  const normalized = normalizeImagePath(sample.image_path);
  const pathToRun = normalized || String(sample.image_path || "");
  const xTargetToRun = Number(sample.x_target);
  if (imgEl && normalized) {
    const imgUrl = `/api/datasets/${encodeURIComponent(SWEEP_DATASET)}/image?path=${encodeURIComponent(normalized)}`;
    imgEl.src = imgUrl;
    imgEl.style.display = "block";
  } else if (imgEl) {
    imgEl.style.display = "none";
  }

  if (detailsActions) {
    detailsActions.style.display = jobId ? "" : "none";
  }
  if (detailsRunResult) {
    detailsRunResult.textContent = "";
  }
  if (jobId && pathToRun && Number.isFinite(xTargetToRun)) {
    detailsRunContext = { imagePath: pathToRun, xTarget: xTargetToRun };
  } else {
    detailsRunContext = null;
  }

  setModalOpen(detailsModal, true);
}

function attachTableInteractions() {
  const table = byId("samplesTable");
  if (!table) return;
  const tbody = table.querySelector("tbody");
  const headers = table.querySelectorAll("th[data-type]");
  let sortState = { index: -1, asc: true };

  headers.forEach((header, colIdx) => {
    header.addEventListener("click", () => {
      const type = header.getAttribute("data-type") || "text";
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const asc = sortState.index === colIdx ? !sortState.asc : true;
      rows.sort((a, b) => {
        const cellA = a.children[colIdx];
        const cellB = b.children[colIdx];
        const valA = (cellA && cellA.getAttribute("data-value")) || (cellA ? cellA.textContent : "");
        const valB = (cellB && cellB.getAttribute("data-value")) || (cellB ? cellB.textContent : "");

        if (type === "number") {
          const numA = Number(valA);
          const numB = Number(valB);
          if (Number.isFinite(numA) && Number.isFinite(numB)) {
            return asc ? numA - numB : numB - numA;
          }
        }
        return asc
          ? String(valA).localeCompare(String(valB))
          : String(valB).localeCompare(String(valA));
      });

      rows.forEach((row) => tbody.appendChild(row));
      sortState = { index: colIdx, asc };
    });
  });

  const searchInput = byId("sampleSearch");
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      const needle = searchInput.value.trim().toLowerCase();
      const rows = tbody.querySelectorAll("tr");
      rows.forEach((row) => {
        const visible = !needle || row.textContent.toLowerCase().includes(needle);
        row.style.display = visible ? "" : "none";
      });
    });
  }

  table.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains("details-btn")) return;
    const index = Number(target.dataset.index);
    if (!Number.isInteger(index) || index < 0) return;
    openSampleDetails(index);
  });
}

function renderDrilldownList(items) {
  const list = byId("drilldownList");
  const status = byId("drilldownStatus");
  if (!list || !status) return;

  list.innerHTML = "";
  status.textContent = `Found ${items.length} samples.`;
  if (!items.length) {
    status.textContent = "No samples found for this target.";
    return;
  }

  const frag = document.createDocumentFragment();
  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "sample-item";

    const textWrap = document.createElement("div");
    textWrap.className = "sample-item__text";

    const link = document.createElement("a");
    const normalized = normalizeImagePath(item.image_path);
    link.textContent = item.image_path || "(unknown image)";
    if (normalized) {
      link.href = `/api/datasets/${encodeURIComponent(SWEEP_DATASET)}/image?path=${encodeURIComponent(normalized)}`;
      link.target = "_blank";
      link.rel = "noopener";
    }

    const meta = document.createElement("div");
    meta.className = "sample-item__meta";
    const timeText = item.decode_time_ms != null ? `${Number(item.decode_time_ms).toFixed(3)} ms` : "n/a";
    const metricKind = String(item.metric_kind || "").trim() || (item.no_gt ? "decode_success_rate" : "gt_accuracy");
    const accText = item.accuracy == null ? "n/a" : Number(item.accuracy).toFixed(3);
    meta.textContent = `decode_success=${item.decode_success ? "yes" : "no"}, ${metricKind}=${accText}, time=${timeText}`;

    textWrap.appendChild(link);
    textWrap.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "sample-item__actions";
    const buttonRow = document.createElement("div");
    buttonRow.className = "sample-item__button-row";

    if (item.no_gt) {
      const badge = document.createElement("span");
      badge.className = "no-gt-badge";
      badge.textContent = "NO GT";
      buttonRow.appendChild(badge);
    }

    const detailsBtn = document.createElement("button");
    detailsBtn.type = "button";
    detailsBtn.className = "details-btn";
    detailsBtn.textContent = "Details";
    detailsBtn.addEventListener("click", () => {
      if (Number.isInteger(item.sample_index)) {
        openSampleDetails(item.sample_index);
      }
    });
    buttonRow.appendChild(detailsBtn);

    actions.appendChild(buttonRow);

    if (jobId) {
      const runBtn = document.createElement("button");
      runBtn.type = "button";
      runBtn.className = "details-btn";
      runBtn.textContent = "Run on this image";

      const runResult = document.createElement("div");
      runResult.className = "sample-item__run-result";

      runBtn.addEventListener("click", () => {
        const normalizedPath = normalizeImagePath(item.image_path);
        const pathToRun = normalizedPath || item.image_path;
        runSingleSample(pathToRun, item.x_target, runBtn, runResult);
      });

      buttonRow.appendChild(runBtn);
      actions.appendChild(runResult);
    }

    li.appendChild(textWrap);
    li.appendChild(actions);
    frag.appendChild(li);
  });

  list.appendChild(frag);
}

async function openTargetSamples(xTarget) {
  const title = byId("drilldownTitle");
  const status = byId("drilldownStatus");
  const list = byId("drilldownList");
  if (!title || !status || !list) return;

  title.textContent = `Samples for module size=${Number(xTarget).toString()} px`;
  status.textContent = "Loading...";
  list.innerHTML = "";
  setModalOpen(drilldownModal, true);

  if (!jobId) {
    const localItems = SWEEP_SAMPLE_DATA
      .map((row) => ({ ...row, sample_index: row.sample_index }))
      .filter((row) => Number(row.x_target) === Number(xTarget));
    renderDrilldownList(localItems);
    return;
  }

  try {
    const resp = await fetch(`/api/jobs/${jobId}/samples?x_target=${encodeURIComponent(xTarget)}&limit=200`);
    const payload = await resp.json();
    if (!resp.ok || !Array.isArray(payload)) {
      status.textContent = payload && payload.error ? payload.error : "Request failed.";
      return;
    }
    renderDrilldownList(payload);
  } catch (err) {
    status.textContent = "Request failed.";
  }
}

function attachPlotClickHandler() {
  const plot = document.querySelector(".plotly-graph-div");
  if (!plot || typeof plot.on !== "function") return;

  plot.on("plotly_click", (event) => {
    const point = event && event.points && event.points[0];
    if (!point) return;
    const traceName = String((point.data && point.data.name) || "").toLowerCase();
    if (traceName.includes("time") || traceName.includes("success") || traceName.includes("accuracy")) {
      const xTarget = Number(point.x);
      if (Number.isFinite(xTarget)) {
        openTargetSamples(xTarget);
      }
    }
  });
}

attachTableInteractions();
attachPlotClickHandler();
"""
    )
    html_parts.append('</script>')

    html_parts.append(
        """
    </div>
  </body>
</html>
"""
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("".join(html_parts), encoding="utf-8")
