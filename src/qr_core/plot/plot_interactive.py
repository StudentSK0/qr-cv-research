from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

from ..binning import BinStats
from ..metrics import SampleResult
from ..pareto import aggregate_module_sizes, pareto_front


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
    quantized = int(round(rounded / step) * step)
    return max(step, quantized)


def _warn_if_time_is_binned(x_time_raw: Sequence[float], step_px: int) -> None:
    if step_px <= 1 or len(x_time_raw) < 2:
        return
    raw_rounded = [int(round(v)) for v in x_time_raw]
    if len(set(raw_rounded)) < 2:
        return
    binned = [_quantize_raw_to_bin(v, step_px) for v in x_time_raw]
    if raw_rounded == binned:
        print(
            "Warning: time plot appears to use binned module sizes. "
            "Ensure raw module_size_raw_px is used for time scatter/trend.",
        )


def _is_no_gt(expected: str | None) -> bool:
    return not str(expected or "").strip()


def _build_baseline_report_html(
    plot_html: str,
    *,
    report_title: str,
    dataset: str,
    engine: str,
    iterations: int,
    time_mode: str,
    total_samples: int,
    gt_samples: int,
    no_gt_samples: int,
    binned_table_rows_html: str,
    pareto_table_rows_html: str,
) -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Baseline report</title>
    <style>
      :root {
        --bg: #eef3f8;
        --card: #ffffff;
        --line: #dbe3ef;
        --ink: #0f172a;
        --muted: #475569;
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
      }
      .table-wrap {
        width: 100%;
        max-height: 520px;
        overflow: auto;
      }
      html.is-embed .baseline-summary-card {
        display: none !important;
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <section class="card baseline-summary-card">
        <h1 style="margin:0 0 10px;">""" + escape(report_title) + """</h1>
        <div class="summary-grid">
          <div class="summary-item"><div class="summary-item__label">Dataset</div><div class="summary-item__value">""" + escape(dataset) + """</div></div>
          <div class="summary-item"><div class="summary-item__label">Engine</div><div class="summary-item__value">""" + escape(engine) + """</div></div>
          <div class="summary-item"><div class="summary-item__label">Iterations</div><div class="summary-item__value">""" + str(int(iterations)) + """</div></div>
          <!-- <div class="summary-item"><div class="summary-item__label">Time mode</div><div class="summary-item__value">""" + escape(time_mode) + """</div></div> -->
          <div class="summary-item"><div class="summary-item__label">Total samples</div><div class="summary-item__value">""" + str(int(total_samples)) + """</div></div>
          <div class="summary-item"><div class="summary-item__label">GT samples</div><div class="summary-item__value">""" + str(int(gt_samples)) + """</div></div>
          <div class="summary-item"><div class="summary-item__label">NO GT samples</div><div class="summary-item__value">""" + str(int(no_gt_samples)) + """</div></div>
        </div>
      </section>
      <section class="card">
""" + plot_html + """
      </section>
      <section class="card">
        <h3 style="margin:0 0 10px;">Pareto summary</h3>
        <div class="table-wrap">
          <table id="baselineParetoSummary">
            <thead>
              <tr>
                <th>pareto module size</th>
                <th>samples</th>
                <th>time min (sec)</th>
                <th>accuracy mean</th>
              </tr>
            </thead>
            <tbody>
""" + pareto_table_rows_html + """
            </tbody>
          </table>
        </div>
      </section>
      <section class="card">
        <h3 style="margin:0 0 10px;">Per-target summary</h3>
        <div class="table-wrap">
          <table id="baselineTargetSummary">
            <thead>
              <tr>
                <th>module size</th>
                <th>samples</th>
                <th>accuracy</th>
                <th>correct decodes</th>
                <th>failed decodes</th>
              </tr>
            </thead>
            <tbody>
""" + binned_table_rows_html + """
            </tbody>
          </table>
        </div>
      </section>
    </div>
    <script>
      (function () {
        try {
          var params = new URLSearchParams(window.location.search || "");
          if (params.get("embed") === "1") {
            document.documentElement.classList.add("is-embed");
          }
        } catch (err) {
          // no-op
        }
      })();
    </script>
  </body>
</html>
"""


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
    y_time_raw = [float(r.time_total_min_sec) * 1000.0 for r in results_sorted]

    x_acc_bin = [b.module_size for b in bins_sorted]
    tick_step = _infer_tick_step(x_acc_bin) if x_acc_bin else 1
    y_count_acc1 = [b.count_acc1 for b in bins_sorted]
    y_count_acc0 = [b.count_acc0 for b in bins_sorted]

    bin_step_px = int(results_sorted[0].module_bin_step_px) if results_sorted else 1
    _warn_if_time_is_binned(x_time_raw, bin_step_px)

    pareto_stats = aggregate_module_sizes(results_sorted, n_min=10)
    pareto_front_points = pareto_front(
        pareto_stats,
        minimize_keys=("time_min",),
        maximize_keys=("accuracy_mean",),
    )
    x_pareto_bins = [p["module_size"] for p in pareto_front_points]
    y_pareto_bins = [float(p["time_min"]) * 1000.0 for p in pareto_front_points]
    pareto_acc = [p["accuracy_mean"] for p in pareto_front_points]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.68, 0.32],
    )

    fig.add_trace(
        go.Scatter(
            x=x_time_raw,
            y=y_time_raw,
            mode="markers",
            name="Time",
            marker=dict(size=6, color="#1f77b4", opacity=0.6),
            customdata=[r.image_path for r in results_sorted],
            hovertemplate="module_size_raw=%{x:.3f}<br>time=%{y:.3f} ms<extra></extra>",
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
            hovertemplate="module_size_raw=%{x:.3f}<br>smoothed_time=%{y:.3f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if x_pareto_bins:
        fig.add_trace(
            go.Scatter(
                x=x_pareto_bins,
                y=y_pareto_bins,
                mode="markers",
                name="Pareto bins",
                marker=dict(size=10, color="#ff7f0e", symbol="diamond", line=dict(width=1)),
                customdata=pareto_acc,
                hovertemplate="module_size_bin=%{x:.0f}<br>time_min=%{y:.3f} ms<br>accuracy_mean=%{customdata:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=x_acc_bin,
            y=y_count_acc1,
            name="Correct decodes",
            marker=dict(color="#2ca02c"),
            hovertemplate="module_size=%{x}<br>correct=%{y}<extra></extra>",
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
            hovertemplate="module_size=%{x}<br>failed=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    max_time = max(y_time_raw, default=0.0)
    max_count = max(y_count_acc1 + y_count_acc0, default=0)
    max_x = max(
        [max(x_time_raw, default=0.0), max(x_acc_bin, default=0), 1]
    )

    fig.update_layout(
        title=title or "QR decoding vs module size",
        barmode="stack",
        hovermode="closest",
        height=860,
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

    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    time_upper = max_time * 1.05 if max_time > 0 else 1.0
    # Slightly extend below zero so near-zero points are visible above the x-axis.
    time_lower = -max(0.5, time_upper * 0.03)
    fig.update_yaxes(range=[time_lower, time_upper], row=1, col=1)

    count_upper = max_count * 1.15 if max_count > 0 else 1.0
    fig.update_yaxes(range=[0, count_upper], row=2, col=1)

    first = results_sorted[0]
    dataset = str(first.dataset)
    engine = str(first.engine)
    iterations = int(first.decode_iterations)
    time_mode = str(first.time_mode)

    total_samples = len(results_sorted)
    no_gt_samples = sum(1 for item in results_sorted if _is_no_gt(item.expected))
    gt_samples = total_samples - no_gt_samples

    report_title = f"QR baseline run ({dataset}, {engine})"

    binned_table_rows: list[str] = []
    for b in bins_sorted:
        binned_table_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{int(b.module_size)}</td>",
                    f"<td>{int(b.count)}</td>",
                    f"<td>{float(b.mean_accuracy * 100.0):.2f}%</td>",
                    f"<td>{int(b.count_acc1)}</td>",
                    f"<td>{int(b.count_acc0)}</td>",
                    "</tr>",
                ]
            )
        )
    if not binned_table_rows:
        binned_table_rows.append('<tr><td colspan="5">No data.</td></tr>')

    pareto_table_rows: list[str] = []
    for p in pareto_front_points:
        pareto_table_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{float(p['module_size']):.0f}</td>",
                    f"<td>{float(p['n']):.0f}</td>",
                    f"<td>{float(p['time_min']):.6f}</td>",
                    f"<td>{float(p['accuracy_mean']):.3f}</td>",
                    "</tr>",
                ]
            )
        )
    if not pareto_table_rows:
        pareto_table_rows.append('<tr><td colspan="4">No Pareto points.</td></tr>')

    plot_html = fig.to_html(full_html=False, include_plotlyjs=True)

    html = _build_baseline_report_html(
        plot_html,
        report_title=report_title,
        dataset=dataset,
        engine=engine,
        iterations=iterations,
        time_mode=time_mode,
        total_samples=total_samples,
        gt_samples=gt_samples,
        no_gt_samples=no_gt_samples,
        binned_table_rows_html="".join(binned_table_rows),
        pareto_table_rows_html="".join(pareto_table_rows),
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
