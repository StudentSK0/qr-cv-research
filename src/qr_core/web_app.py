from __future__ import annotations

import re
import shutil
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template_string, request, url_for

from .binning import build_bin_stats
from .dataset_io import iter_qr_samples
from .engines import ENGINE_REGISTRY
from .metrics import ExperimentConfig, ProgressState, run_experiment, save_results_json
from .plot.plot_interactive import build_interactive_plot


INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>QR CV Research</title>
    <style>
      :root {
        --bg-start: #f5efe6;
        --bg-end: #e6f0f6;
        --ink: #1b1f2a;
        --muted: #516070;
        --accent: #0f8b8d;
        --accent-2: #f4a259;
        --accent-3: #19323c;
        --surface: rgba(255, 255, 255, 0.86);
        --surface-strong: #ffffff;
        --line: rgba(30, 40, 60, 0.12);
        --shadow: 0 30px 60px rgba(15, 20, 30, 0.18);
        --radius: 22px;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Sora", "IBM Plex Sans", "Fira Sans", sans-serif;
        color: var(--ink);
        background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
      }
      body::before,
      body::after {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        filter: blur(0);
        opacity: 0.35;
        z-index: 0;
      }
      body::before {
        top: -120px;
        right: -120px;
        background: radial-gradient(circle, #e8f3ff 0%, rgba(232, 243, 255, 0) 70%);
      }
      body::after {
        bottom: -140px;
        left: -140px;
        background: radial-gradient(circle, #fbe7d3 0%, rgba(251, 231, 211, 0) 70%);
      }
      .page {
        max-width: 1100px;
        margin: 0 auto;
        padding: 52px 24px 70px;
        position: relative;
        z-index: 1;
      }
      .hero {
        display: grid;
        gap: 18px;
        margin-bottom: 32px;
      }
      .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.24em;
        font-size: 0.7rem;
        color: var(--muted);
      }
      .hero h1 {
        margin: 8px 0 12px;
        font-size: clamp(2.2rem, 4vw, 3rem);
        letter-spacing: -0.02em;
      }
      .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 620px;
      }
      .card {
        background: var(--surface-strong);
        border-radius: var(--radius);
        padding: 30px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        animation: rise 0.6s ease-out both;
      }
      .form-grid {
        display: grid;
        grid-template-columns: repeat(12, minmax(0, 1fr));
        gap: 18px;
      }
      .field {
        grid-column: span 4;
        display: grid;
        gap: 8px;
      }
      .field--wide {
        grid-column: 1 / -1;
      }
      label {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.02em;
      }
      select,
      input[type="number"],
      input[type="text"],
      input[type="file"] {
        width: 100%;
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid var(--line);
        font-size: 1rem;
        background: #fff;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }
      input[type="file"] {
        background: #f7fbff;
        border-style: dashed;
      }
      input[type="file"]::file-selector-button {
        border: none;
        background: var(--accent-3);
        color: #fff;
        padding: 8px 12px;
        border-radius: 10px;
        margin-right: 12px;
        font-weight: 600;
        cursor: pointer;
      }
      select:focus,
      input:focus {
        outline: none;
        border-color: rgba(15, 139, 141, 0.55);
        box-shadow: 0 0 0 3px rgba(15, 139, 141, 0.15);
      }
      .hint {
        margin: 0;
        font-size: 0.82rem;
        color: var(--muted);
      }
      .tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        margin-left: 6px;
        border-radius: 50%;
        background: #eef3f8;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        cursor: help;
        position: relative;
      }
      .tip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 140%;
        left: 50%;
        transform: translateX(-50%);
        width: 220px;
        padding: 8px 10px;
        border-radius: 10px;
        background: #1f2937;
        color: #fff;
        font-size: 0.78rem;
        line-height: 1.3;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease, transform 0.15s ease;
        transform-origin: bottom center;
        z-index: 10;
      }
      .tip::before {
        content: "";
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent;
        opacity: 0;
        transition: opacity 0.15s ease;
      }
      .tip:hover::after,
      .tip:focus::after,
      .tip:hover::before,
      .tip:focus::before {
        opacity: 1;
      }
      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        align-items: center;
        margin-top: 22px;
      }
      button {
        border: none;
        padding: 12px 22px;
        border-radius: 999px;
        background: linear-gradient(120deg, var(--accent), #0b6d6f);
        color: #fff;
        font-size: 1rem;
        cursor: pointer;
        box-shadow: 0 12px 22px rgba(15, 139, 141, 0.28);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(15, 139, 141, 0.32);
      }
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: var(--muted);
      }
      .badge span {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--accent-2);
      }
      .error {
        margin-top: 18px;
        padding: 12px 16px;
        border-radius: 14px;
        background: #fff3ef;
        color: #a04632;
        border: 1px solid #f2c7ba;
      }
      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      @media (max-width: 980px) {
        .hero {
          grid-template-columns: 1fr;
        }
      }
      @media (max-width: 720px) {
        .field {
          grid-column: span 6;
        }
      }
      @media (max-width: 600px) {
        .field {
          grid-column: 1 / -1;
        }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <header class="hero">
        <div class="hero__text">
          <span class="eyebrow">Experiment console</span>
          <h1>QR CV Research</h1>
          <p>Run decoding experiments across engines and datasets to measure time and accuracy vs module size.</p>
        </div>
      </header>
      <section class="card">
        <form method="post" action="{{ url_for('run_experiment_route') }}" enctype="multipart/form-data">
          <div class="form-grid">
            <div class="field field--wide">
              <label for="dataset_file">Dataset zip</label>
              <input id="dataset_file" name="dataset_file" type="file" accept=".zip" required>
              <p class="hint">Zip root: images/QR_CODE and markup/QR_CODE. If the name already exists, the existing dataset is reused.</p>
            </div>
            <div class="field">
              <label for="engine">Engine</label>
              <select id="engine" name="engine" required>
                {% for engine in engines %}
                <option value="{{ engine }}">{{ engine }}</option>
                {% endfor %}
              </select>
            </div>
            <div class="field">
              <label for="iterations">Iterations
                <span class="tip" tabindex="0" data-tip="Number of decode repeats per image. time_total uses the minimum across repeats.">?</span>
              </label>
              <input id="iterations" name="iterations" type="number" min="1" value="3" required>
            </div>
            <div class="field">
              <label for="bin_step_px">Module size bin step (px)
                <span class="tip" tabindex="0" data-tip="Bin width for module_size in pixels. Samples are grouped by this step.">?</span>
              </label>
              <input id="bin_step_px" name="bin_step_px" type="number" min="1" value="2" required>
            </div>
          </div>
          <div class="actions">
            <button type="submit">Run experiment</button>
            <div class="badge"><span></span>time_total min over iterations</div>
          </div>
          {% if error %}
          <div class="error">{{ error }}</div>
          {% endif %}
        </form>
      </section>
    </div>
  </body>
</html>
"""

PROGRESS_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>QR CV Research - Running</title>
    <style>
      :root {
        --bg-start: #f5efe6;
        --bg-end: #e6f0f6;
        --ink: #1b1f2a;
        --muted: #516070;
        --accent: #0f8b8d;
        --accent-2: #f4a259;
        --accent-3: #19323c;
        --surface: rgba(255, 255, 255, 0.88);
        --surface-strong: #ffffff;
        --line: rgba(30, 40, 60, 0.12);
        --shadow: 0 26px 50px rgba(15, 20, 30, 0.16);
        --radius: 22px;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Sora", "IBM Plex Sans", "Fira Sans", sans-serif;
        color: var(--ink);
        background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
      }
      body::before,
      body::after {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        opacity: 0.3;
        z-index: 0;
      }
      body::before {
        top: -120px;
        right: -120px;
        background: radial-gradient(circle, #e8f3ff 0%, rgba(232, 243, 255, 0) 70%);
      }
      body::after {
        bottom: -120px;
        left: -120px;
        background: radial-gradient(circle, #fbe7d3 0%, rgba(251, 231, 211, 0) 70%);
      }
      .page {
        max-width: 920px;
        margin: 0 auto;
        padding: 42px 24px 64px;
        display: grid;
        gap: 24px;
        position: relative;
        z-index: 1;
      }
      .card {
        background: var(--surface-strong);
        border-radius: var(--radius);
        padding: 26px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
        color: var(--muted);
        font-size: 0.92rem;
      }
      .meta__item {
        padding: 6px 10px;
        border-radius: 999px;
        background: #f3f6fb;
        border: 1px solid rgba(30, 40, 60, 0.08);
      }
      .progress {
        position: relative;
        height: 16px;
        border-radius: 999px;
        background: rgba(15, 25, 40, 0.08);
        overflow: hidden;
        margin-top: 18px;
      }
      .progress__fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, var(--accent), #19a6a0, var(--accent));
        background-size: 200% 100%;
        transition: width 0.4s ease;
        animation: flow 1.8s linear infinite;
      }
      .progress__meta {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        color: var(--muted);
        font-size: 0.9rem;
      }
      .summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 18px;
      }
      .summary div {
        padding: 12px 14px;
        border-radius: 14px;
        background: #f6f8fb;
        color: var(--muted);
        font-size: 0.9rem;
      }
      .tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        margin-left: 6px;
        border-radius: 50%;
        background: #eef3f8;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        cursor: help;
        position: relative;
      }
      .tip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 140%;
        left: 50%;
        transform: translateX(-50%);
        width: 220px;
        padding: 8px 10px;
        border-radius: 10px;
        background: #1f2937;
        color: #fff;
        font-size: 0.78rem;
        line-height: 1.3;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease, transform 0.15s ease;
        transform-origin: bottom center;
        z-index: 10;
      }
      .tip::before {
        content: "";
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent;
        opacity: 0;
        transition: opacity 0.15s ease;
      }
      .tip:hover::after,
      .tip:focus::after,
      .tip:hover::before,
      .tip:focus::before {
        opacity: 1;
      }
      .tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        margin-left: 6px;
        border-radius: 50%;
        background: #eef3f8;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        cursor: help;
        position: relative;
      }
      .tip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 140%;
        left: 50%;
        transform: translateX(-50%);
        width: 220px;
        padding: 8px 10px;
        border-radius: 10px;
        background: #1f2937;
        color: #fff;
        font-size: 0.78rem;
        line-height: 1.3;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease, transform 0.15s ease;
        transform-origin: bottom center;
        z-index: 10;
      }
      .tip::before {
        content: "";
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent;
        opacity: 0;
        transition: opacity 0.15s ease;
      }
      .tip:hover::after,
      .tip:focus::after,
      .tip:hover::before,
      .tip:focus::before {
        opacity: 1;
      }
      .error {
        margin-top: 16px;
        padding: 12px 16px;
        border-radius: 14px;
        background: #fff3ef;
        color: #a04632;
        border: 1px solid #f2c7ba;
      }
      @keyframes flow {
        from {
          background-position: 0% 50%;
        }
        to {
          background-position: 100% 50%;
        }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <div class="card">
        <h1>Experiment running</h1>
        <div class="meta">
          <span class="meta__item">Dataset: <strong>{{ dataset }}</strong></span>
          <span class="meta__item">Engine: <strong>{{ engine }}</strong></span>
        </div>
        <div class="progress">
          <div class="progress__fill" id="progress-fill"></div>
        </div>
        <div class="progress__meta">
          <span id="progress-text">0%</span>
          <span id="progress-count">0 / {{ total }}</span>
        </div>
        <div class="summary">
          <div>processed: <span id="processed">0</span></div>
          <div>skipped_no_markup: <span id="skipped_no_markup">0</span>
            <span class="tip" tabindex="0" data-tip="Images skipped because markup JSON is missing or unreadable.">?</span>
          </div>
        </div>
        <div class="error" id="error-box" style="display:none;"></div>
      </div>
    </div>
    <script>
      const jobId = "{{ job_id }}";
      const total = Number("{{ total }}");
      const progressFill = document.getElementById("progress-fill");
      const progressText = document.getElementById("progress-text");
      const progressCount = document.getElementById("progress-count");
      const processedEl = document.getElementById("processed");
      const skippedNoMarkupEl = document.getElementById("skipped_no_markup");
      const errorBox = document.getElementById("error-box");

      async function poll() {
        try {
          const response = await fetch(`/progress/${jobId}`);
          if (!response.ok) {
            throw new Error(`Progress request failed (${response.status})`);
          }
          const data = await response.json();
          const seen = data.seen || 0;
          const percent = total > 0 ? Math.round((seen / total) * 100) : 0;
          progressFill.style.width = `${percent}%`;
          progressText.textContent = `${percent}%`;
          progressCount.textContent = `${seen} / ${total}`;
          processedEl.textContent = data.processed || 0;
          skippedNoMarkupEl.textContent = data.skipped_no_markup || 0;

          if (data.status === "done") {
            window.location.href = `/results/${jobId}`;
            return;
          }
          if (data.status === "error") {
            errorBox.textContent = data.error || "Experiment failed.";
            errorBox.style.display = "block";
            return;
          }
        } catch (error) {
          errorBox.textContent = error.message;
          errorBox.style.display = "block";
          return;
        }
        setTimeout(poll, 1000);
      }

      poll();
    </script>
  </body>
</html>
"""

RESULT_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>QR CV Research - Results</title>
    <style>
      :root {
        --bg-start: #f5efe6;
        --bg-end: #e6f0f6;
        --ink: #1b1f2a;
        --muted: #516070;
        --accent: #0f8b8d;
        --accent-2: #f4a259;
        --accent-3: #19323c;
        --surface: rgba(255, 255, 255, 0.88);
        --surface-strong: #ffffff;
        --line: rgba(30, 40, 60, 0.12);
        --shadow: 0 26px 50px rgba(15, 20, 30, 0.16);
        --radius: 22px;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Sora", "IBM Plex Sans", "Fira Sans", sans-serif;
        color: var(--ink);
        background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
      }
      body::before,
      body::after {
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        border-radius: 50%;
        opacity: 0.3;
        z-index: 0;
      }
      body::before {
        top: -120px;
        right: -120px;
        background: radial-gradient(circle, #e8f3ff 0%, rgba(232, 243, 255, 0) 70%);
      }
      body::after {
        bottom: -120px;
        left: -120px;
        background: radial-gradient(circle, #fbe7d3 0%, rgba(251, 231, 211, 0) 70%);
      }
      .page {
        max-width: 1100px;
        margin: 0 auto;
        padding: 40px 24px 64px;
        display: grid;
        gap: 24px;
        position: relative;
        z-index: 1;
      }
      .card {
        background: var(--surface-strong);
        border-radius: var(--radius);
        padding: 26px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
      }
      .back-link {
        text-decoration: none;
        color: var(--muted);
        font-size: 0.9rem;
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
        color: var(--muted);
        font-size: 0.92rem;
      }
      .meta__item {
        padding: 6px 10px;
        border-radius: 999px;
        background: #f3f6fb;
        border: 1px solid rgba(30, 40, 60, 0.08);
      }
      .summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 16px;
      }
      .summary div {
        padding: 12px 14px;
        border-radius: 14px;
        background: #f6f8fb;
        color: var(--muted);
        font-size: 0.9rem;
      }
      .links {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 14px;
      }
      .links a {
        text-decoration: none;
        color: #fff;
        background: var(--accent-3);
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 600;
      }
      .links a:last-child {
        background: var(--accent);
      }
      iframe {
        width: 100%;
        height: 820px;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #fff;
      }
      .tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        margin-left: 6px;
        border-radius: 50%;
        background: #eef3f8;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        cursor: help;
        position: relative;
      }
      .tip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 140%;
        left: 50%;
        transform: translateX(-50%);
        width: 220px;
        padding: 8px 10px;
        border-radius: 10px;
        background: #1f2937;
        color: #fff;
        font-size: 0.78rem;
        line-height: 1.3;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease, transform 0.15s ease;
        transform-origin: bottom center;
        z-index: 10;
      }
      .tip::before {
        content: "";
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent;
        opacity: 0;
        transition: opacity 0.15s ease;
      }
      .tip:hover::after,
      .tip:focus::after,
      .tip:hover::before,
      .tip:focus::before {
        opacity: 1;
      }
    </style>
  </head>
  <body>
    <div class="page">
      <div class="card">
        <a class="back-link" href="{{ url_for('index') }}">Back to setup</a>
        <h1>Experiment results</h1>
        <div class="meta">
          <span class="meta__item">Dataset: <strong>{{ dataset }}</strong></span>
          <span class="meta__item">Engine: <strong>{{ engine }}</strong></span>
          <span class="meta__item">Iterations: <strong>{{ iterations }}</strong></span>
          <span class="meta__item">Bin step: <strong>{{ bin_step_px }} px</strong></span>
          <span class="meta__item">Mode: <strong>time_total min</strong></span>
        </div>
        <div class="summary">
          <div>processed: {{ summary.processed }}</div>
          <div>skipped_no_markup: {{ summary.skipped_no_markup }}
            <span class="tip" tabindex="0" data-tip="Images skipped because markup JSON is missing or unreadable.">?</span>
          </div>
        </div>
        <div class="links">
          <a href="{{ plot_url }}" target="_blank" rel="noopener">Open plot</a>
          <a href="{{ json_url }}" target="_blank" rel="noopener">Download JSON</a>
        </div>
      </div>
      <div class="card">
        <iframe src="{{ plot_url }}"></iframe>
      </div>
    </div>
  </body>
</html>
"""


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _is_project_root(path: Path) -> bool:
    return (path / "datasets").is_dir() and (path / "src").is_dir()


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if _is_project_root(cwd):
        return cwd

    fallback = Path(__file__).resolve().parents[2]
    if _is_project_root(fallback):
        return fallback

    raise RuntimeError("Cannot locate project root. Expected datasets/ and src/.")


def _parse_positive_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _safe_engine_key(engine_key: str, registry: dict[str, Any]) -> bool:
    return engine_key in registry


def _safe_dataset_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cleaned):
        return None
    return cleaned


def _has_dataset_structure(path: Path) -> bool:
    return (path / "images" / "QR_CODE").is_dir() and (path / "markup" / "QR_CODE").is_dir()


def _find_dataset_root(extracted_dir: Path) -> Path | None:
    if _has_dataset_structure(extracted_dir):
        return extracted_dir

    candidates = [d for d in extracted_dir.iterdir() if d.is_dir() and d.name != "__MACOSX"]
    if len(candidates) == 1 and _has_dataset_structure(candidates[0]):
        return candidates[0]

    return None


def _safe_extract_zip(zip_file: zipfile.ZipFile, dest_dir: Path) -> None:
    for member in zip_file.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("Unsafe path in zip archive.")
    zip_file.extractall(dest_dir)


def _store_uploaded_dataset(
    file_storage: Any,
    project_root: Path,
) -> tuple[str | None, str | None]:
    if file_storage is None or not getattr(file_storage, "filename", ""):
        return None, "No dataset zip provided."

    safe_name = _safe_dataset_name(Path(file_storage.filename).stem)

    if safe_name is None:
        return None, "Invalid dataset name derived from zip filename. Use letters, numbers, dash, underscore."

    dataset_dir = project_root / "datasets" / safe_name
    if dataset_dir.exists():
        if _has_dataset_structure(dataset_dir):
            return safe_name, None
        return None, f"Dataset '{safe_name}' exists but has invalid structure."

    uploads_dir = project_root / "datasets" / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = uploads_dir / f"tmp_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_storage.stream.seek(0)
        with zipfile.ZipFile(file_storage.stream) as zf:
            _safe_extract_zip(zf, temp_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, "Uploaded file is not a valid zip archive."
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, f"Failed to unpack zip: {exc}"

    candidate_root = _find_dataset_root(temp_dir)
    if candidate_root is None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, "Zip must contain images/QR_CODE and markup/QR_CODE."

    try:
        dataset_dir.mkdir(parents=True, exist_ok=False)
        for item in candidate_root.iterdir():
            shutil.move(str(item), dataset_dir)
    except Exception as exc:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, f"Failed to store dataset: {exc}"

    shutil.rmtree(temp_dir, ignore_errors=True)

    if not _has_dataset_structure(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
        return None, "Dataset structure invalid after extraction."

    return safe_name, None


def _init_job(
    job_id: str,
    dataset: str,
    engine_key: str,
    iterations: int,
    bin_step_px: int,
    total: int,
) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "dataset": dataset,
            "engine": engine_key,
            "iterations": iterations,
            "bin_step_px": bin_step_px,
            "total": total,
            "seen": 0,
            "processed": 0,
            "skipped_no_markup": 0,
            "error": None,
            "json_file": None,
            "html_file": None,
            "summary": None,
        }


def _update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)


def _get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _run_job(
    job_id: str,
    project_root: Path,
    dataset: str,
    engine_key: str,
    iterations: int,
    bin_step_px: int,
) -> None:
    _update_job(job_id, status="running")

    try:
        engine = ENGINE_REGISTRY[engine_key]()
    except Exception as exc:
        _update_job(job_id, status="error", error=f"Failed to initialize engine: {exc}")
        return

    cfg = ExperimentConfig(
        dataset_name=dataset,
        decode_iterations=iterations,
        module_bin_step_px=bin_step_px,
        time_mode="time_total_min",
    )

    def _progress_cb(state: ProgressState) -> None:
        _update_job(
            job_id,
            seen=state.seen,
            processed=state.processed,
            skipped_no_markup=state.skipped_no_markup,
        )

    try:
        results, summary = run_experiment(project_root, engine, cfg, progress_cb=_progress_cb)
    except Exception as exc:
        _update_job(job_id, status="error", error=f"Experiment failed: {exc}")
        return

    if not results:
        _update_job(job_id, status="error", error="No results produced.")
        return

    out_json = save_results_json(project_root, engine.name, dataset, results)
    bin_stats = build_bin_stats(results)
    out_html = out_json.with_name(f"qr_experiment_plot_{dataset}_{engine.name}.html")
    build_interactive_plot(
        results,
        bin_stats,
        out_html,
        title=f"QR decoding vs module size ({dataset}, {engine.name})",
    )

    _update_job(
        job_id,
        status="done",
        json_file=out_json.name,
        html_file=out_html.name,
        summary={
            "processed": summary.processed,
            "skipped_no_markup": summary.skipped_no_markup,
        },
    )


def create_app(project_root: Path | None = None) -> Flask:
    root = project_root or _resolve_project_root()
    app = Flask(__name__)
    app.config["PROJECT_ROOT"] = root

    @app.get("/")
    def index() -> str:
        engines = list(ENGINE_REGISTRY.keys())
        error = None
        if not engines:
            error = "No engines registered."
        return render_template_string(
            INDEX_TEMPLATE,
            engines=engines,
            error=error,
        )

    @app.post("/run")
    def run_experiment_route() -> str:
        engines = list(ENGINE_REGISTRY.keys())

        iterations = _parse_positive_int(request.form.get("iterations"), default=3)
        bin_step_px = _parse_positive_int(request.form.get("bin_step_px"), default=2)
        engine_key = request.form.get("engine", "")

        if not _safe_engine_key(engine_key, ENGINE_REGISTRY):
            return render_template_string(
                INDEX_TEMPLATE,
                engines=engines,
                error="Unknown engine selected.",
            )

        dataset_file = request.files.get("dataset_file")
        dataset, error = _store_uploaded_dataset(dataset_file, root)
        if error:
            return render_template_string(
                INDEX_TEMPLATE,
                engines=engines,
                error=error,
            )

        total = sum(1 for _ in iter_qr_samples(root, dataset))
        if total == 0:
            return render_template_string(
                INDEX_TEMPLATE,
                engines=engines,
                error="No images found in dataset.",
            )

        job_id = uuid.uuid4().hex
        _init_job(job_id, dataset, engine_key, iterations, bin_step_px, total)

        thread = threading.Thread(
            target=_run_job,
            args=(job_id, root, dataset, engine_key, iterations, bin_step_px),
            daemon=True,
        )
        thread.start()

        return redirect(url_for("job_status", job_id=job_id))

    @app.get("/jobs/<job_id>")
    def job_status(job_id: str) -> str:
        job = _get_job(job_id)
        if job is None:
            abort(404)
        return render_template_string(
            PROGRESS_TEMPLATE,
            job_id=job_id,
            dataset=job["dataset"],
            engine=job["engine"],
            total=job["total"],
        )

    @app.get("/progress/<job_id>")
    def progress(job_id: str):
        job = _get_job(job_id)
        if job is None:
            abort(404)
        return jsonify(
            {
                "status": job["status"],
                "seen": job["seen"],
                "processed": job["processed"],
                "skipped_no_markup": job["skipped_no_markup"],
                "total": job["total"],
                "error": job.get("error"),
            }
        )

    @app.get("/results/<job_id>")
    def results(job_id: str) -> str:
        job = _get_job(job_id)
        if job is None:
            abort(404)
        if job.get("status") != "done":
            return redirect(url_for("job_status", job_id=job_id))

        if not job.get("html_file") or not job.get("json_file"):
            abort(500)

        plot_url = url_for("serve_output", engine=job["engine"], filename=job["html_file"])
        json_url = url_for("serve_output", engine=job["engine"], filename=job["json_file"])

        return render_template_string(
            RESULT_TEMPLATE,
            dataset=job["dataset"],
            engine=job["engine"],
            iterations=job["iterations"],
            bin_step_px=job["bin_step_px"],
            summary=job["summary"],
            plot_url=plot_url,
            json_url=json_url,
        )

    @app.get("/outputs/<engine>/<path:filename>")
    def serve_output(engine: str, filename: str):
        outputs_dir = (root / "outputs" / f"{engine}_json_and_graphics").resolve()
        file_path = (outputs_dir / filename).resolve()

        if not file_path.is_file():
            abort(404)
        if not file_path.is_relative_to(outputs_dir):
            abort(404)

        return file_path.read_bytes(), 200, {"Content-Type": _content_type(file_path)}

    return app


def _content_type(path: Path) -> str:
    if path.suffix.lower() == ".html":
        return "text/html; charset=utf-8"
    if path.suffix.lower() == ".json":
        return "application/json"
    return "application/octet-stream"


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    main()
