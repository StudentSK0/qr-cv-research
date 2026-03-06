from __future__ import annotations

import json
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..metrics import NormalizationSweepSummary, NormalizedSampleResult


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
        success = [1.0 if bool(r.decode_success) else 0.0 for r in rows]
        no_gt_count = sum(1 for r in rows if _is_no_gt(r.expected))
        gt_count = len(rows) - no_gt_count

        stats.append(
            {
                "x_target": float(x_target),
                "count": len(rows),
                "gt_count": gt_count,
                "no_gt_count": no_gt_count,
                "success_rate": float(mean(success) * 100.0) if success else 0.0,
                "time_mean_ms": float(mean(times)) if times else 0.0,
                "time_median_ms": float(median(times)) if times else 0.0,
                "time_p10_ms": _percentile(times, 10.0),
                "time_p90_ms": _percentile(times, 90.0),
                "time_p95_ms": _percentile(times, 95.0),
            }
        )

    return stats


def _build_plot_html(target_stats: Sequence[dict[str, Any]], title: str) -> str:
    x = [row["x_target"] for row in target_stats]
    y_time_median = [row["time_median_ms"] for row in target_stats]
    y_time_mean = [row["time_mean_ms"] for row in target_stats]
    y_time_p10 = [row["time_p10_ms"] for row in target_stats]
    y_time_p90 = [row["time_p90_ms"] for row in target_stats]
    y_success = [row["success_rate"] for row in target_stats]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
        subplot_titles=(
            "Decode time by normalized module size target",
            "Decode success rate by normalized module size target",
        ),
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
            hovertemplate="x_target=%{x:g}px<br>p10=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_time_median,
            mode="lines+markers",
            name="Median decode time",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=8),
            hovertemplate="x_target=%{x:g}px<br>median=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_time_mean,
            mode="lines+markers",
            name="Mean decode time",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=7),
            hovertemplate="x_target=%{x:g}px<br>mean=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_success,
            mode="lines+markers",
            name="Decode success rate",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=8),
            hovertemplate="x_target=%{x:g}px<br>success=%{y:.2f}%<extra></extra>",
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
    fig.update_yaxes(title_text="Decode time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Success rate (%)", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="x_target (px)", row=2, col=1)

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

    summary_by_target: dict[float, Any] = {
        float(row.x_target): row for row in summary.targets
    }

    x_targets_text = ", ".join(f"{float(x):g}" for x in sorted(set(float(x) for x in x_targets)))

    target_summary_rows = []
    for row in target_stats:
        x_target = float(row["x_target"])
        target_info = summary_by_target.get(x_target)

        processed = int(getattr(target_info, "processed", row["count"]))
        skipped_no_markup = int(getattr(target_info, "skipped_no_markup", 0))
        skipped_bad_module = int(getattr(target_info, "skipped_bad_module", 0))
        skipped_image_read = int(getattr(target_info, "skipped_image_read", 0))

        target_summary_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{x_target:g}</td>",
                    f"<td>{int(row['count'])}</td>",
                    f"<td>{row['success_rate']:.2f}%</td>",
                    f"<td>{row['time_median_ms']:.3f}</td>",
                    f"<td>{row['time_mean_ms']:.3f}</td>",
                    f"<td>{row['time_p90_ms']:.3f}</td>",
                    f"<td>{row['time_p95_ms']:.3f}</td>",
                    f"<td>{int(row['gt_count'])}</td>",
                    f"<td>{int(row['no_gt_count'])}</td>",
                    f"<td>{processed}</td>",
                    f"<td>{skipped_no_markup}</td>",
                    f"<td>{skipped_bad_module}</td>",
                    f"<td>{skipped_image_read}</td>",
                    "</tr>",
                ]
            )
        )

    sample_data: list[dict[str, Any]] = []
    table_rows: list[str] = []

    for idx, item in enumerate(sorted_results):
        no_gt = _is_no_gt(item.expected)
        accuracy_value: float | None = None if no_gt else float(item.accuracy)

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
            "accuracy": accuracy_value,
            "no_gt": no_gt,
        }
        sample_data.append(sample_payload)

        expected_display = "N/A (no ground-truth)" if no_gt else _truncate(expected_text)
        accuracy_display = "N/A" if no_gt else f"{float(item.accuracy):.3f}"
        no_gt_badge = '<span class="no-gt-badge">NO GT</span>' if no_gt else ""
        success_display = "yes" if item.decode_success else "no"

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
                    f'<td data-value="{1 if item.decode_success else 0}">{success_display}</td>',
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
      }
      th, td {
        border: 1px solid var(--line);
        padding: 8px;
        text-align: left;
        vertical-align: top;
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
        align-items: center;
        gap: 8px;
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
                f'<div class="summary-item"><div class="summary-item__label">Time mode</div><div class="summary-item__value">{escape(time_mode)}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">module sizes <span class="summary-help" title="Module sizes are target QR module sizes in pixels. The whole dataset is normalized to the first module size, then to each next size, and one combined summary plot is built.">?</span></div><div class="summary-item__value">{escape(x_targets_text)}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">Total samples</div><div class="summary-item__value">{total_samples}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">GT samples</div><div class="summary-item__value">{gt_samples}</div></div>',
                f'<div class="summary-item"><div class="summary-item__label">NO GT samples</div><div class="summary-item__value">{no_gt_samples}</div></div>',
                "</div>",
                '<div class="explain">P10-P90 band is the percentile range from the 10th to the 90th percentile of decode time; it shows the typical spread without extreme outliers.</div>',
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
                '<th>x_target</th><th>samples</th><th>success</th><th>median ms</th><th>mean ms</th><th>p90 ms</th><th>p95 ms</th><th>GT</th><th>NO GT</th><th>processed</th><th>skip no markup</th><th>skip bad module</th><th>skip image read</th>',
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
                '<input id="sampleSearch" type="search" placeholder="Search image_path / decoded / expected...">',
                '<span class="small">Click headers to sort. Accuracy is N/A for NO GT samples.</span>',
                '</div>',
                '<div class="table-wrap">',
                '<table id="samplesTable">',
                '<thead><tr>',
                '<th data-type="number">#</th>',
                '<th data-type="text">image_path</th>',
                '<th data-type="number">x_target</th>',
                '<th data-type="number">module_size_px_current</th>',
                '<th data-type="number">scale_factor</th>',
                '<th data-type="text">resized</th>',
                '<th data-type="number">decode_time_ms</th>',
                '<th data-type="text">decode_success</th>',
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
            <div id="detailsNoGt" class="modal__note" style="display:none;">Ground-truth is not provided for this image. Accuracy comparison is skipped.</div>
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
const drilldownModal = byId("drilldownModal");
const drilldownClose = byId("drilldownClose");

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

async function openSampleDetails(index) {
  const sample = await fetchSampleDetails(index);
  if (!sample) {
    return;
  }

  byId("detailsTitle").textContent = `Sample #${index + 1}: ${sample.image_path || ""}`;
  byId("detailsDecoded").textContent = sample.decoded || "";
  const expectedText = sample.no_gt ? "N/A (no ground-truth)" : (sample.expected || "");
  byId("detailsExpected").textContent = expectedText;

  const accuracyText = sample.no_gt || sample.accuracy == null ? "N/A" : Number(sample.accuracy).toFixed(3);
  const successText = sample.decode_success ? "yes" : "no";
  const meta = [
    `<div><strong>x_target</strong><br>${sample.x_target ?? ""}</div>`,
    `<div><strong>decode_time_ms</strong><br>${sample.decode_time_ms != null ? Number(sample.decode_time_ms).toFixed(3) : ""}</div>`,
    `<div><strong>decode_success</strong><br>${successText}</div>`,
    `<div><strong>accuracy</strong><br>${accuracyText}</div>`,
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
  if (imgEl && normalized) {
    const imgUrl = `/api/datasets/${encodeURIComponent(SWEEP_DATASET)}/image?path=${encodeURIComponent(normalized)}`;
    imgEl.src = imgUrl;
    imgEl.style.display = "block";
  } else if (imgEl) {
    imgEl.style.display = "none";
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
    const accText = item.no_gt || item.accuracy == null ? "N/A" : Number(item.accuracy).toFixed(3);
    meta.textContent = `decode_success=${item.decode_success ? "yes" : "no"}, accuracy=${accText}, time=${timeText}`;

    textWrap.appendChild(link);
    textWrap.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "sample-item__actions";

    if (item.no_gt) {
      const badge = document.createElement("span");
      badge.className = "no-gt-badge";
      badge.textContent = "NO GT";
      actions.appendChild(badge);
    }

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "details-btn";
    btn.textContent = "Details";
    btn.addEventListener("click", () => {
      if (Number.isInteger(item.sample_index)) {
        openSampleDetails(item.sample_index);
      }
    });

    actions.appendChild(btn);

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

  title.textContent = `Samples for x_target=${Number(xTarget).toString()} px`;
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
    if (traceName.includes("time") || traceName.includes("success")) {
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
