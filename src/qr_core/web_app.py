from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from collections import Counter
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template_string, request, url_for

from .binning import build_bin_stats
from .dataset_io import iter_qr_samples
from .engines import ENGINE_REGISTRY
from .engines.base import EngineError
from .metrics import (
    ExperimentConfig,
    ProgressState,
    run_experiment,
    run_module_size_normalization_sweep,
    save_normalization_sweep_json,
    save_results_json,
)
from .markup import extract_expected_value, read_markup
from .plot.plot_interactive import build_interactive_plot
from .plot.plot_sweep_report import build_sweep_report


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
      .mode-switch {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .mode-switch__option {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: #f6f8fb;
        font-size: 0.9rem;
        color: var(--ink);
        cursor: pointer;
      }
      .mode-switch__option input {
        margin: 0;
      }
      .sweep-field {
        display: none;
      }
      .sweep-field.open {
        display: grid;
      }
      .selected-dataset {
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: #f7fbff;
        color: var(--ink);
        min-height: 48px;
        display: flex;
        align-items: center;
        font-weight: 600;
      }
      .dataset-actions {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
      }
      .secondary-button {
        background: #eef3f8;
        color: var(--ink);
        box-shadow: none;
        border: 1px solid var(--line);
      }
      .secondary-button:hover,
      .secondary-button:focus {
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(15, 139, 141, 0.18);
      }
      #browse_zip_btn {
        box-shadow: none;
      }
      #browse_zip_btn:hover,
      #browse_zip_btn:focus {
        box-shadow: 0 16px 28px rgba(15, 139, 141, 0.32);
        transform: translateY(-1px);
      }
      .existing-picker {

        display: none;
        margin-top: 12px;
        gap: 8px;
      }
      .existing-picker.open {
        display: grid;
      }
      .visually-hidden {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
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
      .icon-button {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        border: 1px solid var(--line);
        background: #eef3f8;
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        box-shadow: none;
        cursor: pointer;
      }
      .icon-button:hover,
      .icon-button:focus {
        background: #e3edf6;
        box-shadow: none;
        transform: translateY(-1px);
      }
      .info-modal {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        background: rgba(15, 20, 30, 0.55);
        z-index: 2000;
        padding: 24px;
      }
      .info-modal.open {
        display: flex;
      }
      .info-modal__content {
        background: var(--surface-strong);
        border-radius: 20px;
        padding: 22px;
        width: min(560px, 92vw);
        box-shadow: var(--shadow);
        border: 1px solid var(--line);
      }
      .info-modal__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
      }
      .info-modal__title {
        font-size: 1.1rem;
        font-weight: 700;
      }
      .info-modal__close {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 1px solid var(--line);
        background: #f1f4f8;
        color: var(--muted);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: none;
        padding: 0;
      }
      .info-modal__close:hover {
        background: #e7edf4;
      }
      .info-modal__body {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.5;
      }
      .info-modal__body pre {
        background: #f7fbff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
        margin: 12px 0 0;
        overflow-x: auto;
        color: #1b1f2a;
        font-size: 0.85rem;
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

      .file-picker {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px dashed var(--line);
        background: #f7fbff;
        min-height: 52px;
        justify-content: flex-start;
      }
      .file-picker input[type="file"] {
        position: absolute;
        opacity: 0;
        pointer-events: none;
        width: 0;
        height: 0;
      }
      .file-picker__button {
        border: none;
        background: var(--accent-3);
        color: #fff;
        padding: 10px 14px;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        line-height: 1;
      }
      .file-picker__label {
        color: var(--muted);
        font-size: 0.95rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
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
        <form id="run_form" method="post" action="{{ url_for('run_experiment_route') }}" enctype="multipart/form-data">

          <div class="form-grid">
            <div class="field field--wide">
              <label for="selected_dataset_display">Selected dataset</label>
              <div id="selected_dataset_display" class="selected-dataset" aria-live="polite"></div>
              <input type="hidden" id="dataset_source" name="dataset_source" value="existing">
              <input type="hidden" id="dataset_name" name="dataset_name" value="{{ last_dataset or '' }}">
              <input id="dataset_file" name="dataset_file" type="file" accept=".zip" class="visually-hidden">
            </div>

            <div class="field field--wide">
              <div class="dataset-actions">
                <button id="use_existing_btn" type="button" class="secondary-button">Use existing dataset</button>
                <button id="browse_zip_btn" type="button" class="primary-button">Browse ZIP</button>
                <button id="datasetStructureBtn" type="button" class="icon-button" aria-label="Dataset structure" onclick="(function(){var m=document.getElementById('datasetStructureModal'); if(m){m.classList.add('open'); m.setAttribute('aria-hidden','false');}})()">i</button>
              </div>
              <div id="existing_picker" class="existing-picker" aria-hidden="true">
                <label for="dataset_select">Choose existing dataset</label>
                <select id="dataset_select">
                  {% if available_datasets %}
                  {% for ds in available_datasets %}
                  <option value="{{ ds }}" {% if last_dataset == ds %}selected{% endif %}>{{ ds }}</option>
                  {% endfor %}
                  {% else %}
                  <option value="" disabled>No dataset has been uploaded yet</option>
                  {% endif %}
                </select>
              </div>
            </div>

            <div class="field field--wide">
              <label>Benchmark mode</label>
              <div class="mode-switch" role="radiogroup" aria-label="Benchmark mode">
                <label class="mode-switch__option">
                  <input type="radio" name="run_mode" value="baseline" {% if run_mode != 'sweep' %}checked{% endif %}>
                  <span>source dataset</span>
                </label>
                <label class="mode-switch__option">
                  <input type="radio" name="run_mode" value="sweep" {% if run_mode == 'sweep' %}checked{% endif %}>
                  <span>dataset with a single module size</span>
                </label>
              </div>
            </div>

            <div id="x_targets_field" class="field field--wide sweep-field {% if run_mode == 'sweep' %}open{% endif %}">
              <label for="x_targets">module sizes (px)
                <span class="tip" tabindex="0" data-tip="Module sizes are target QR module sizes in pixels. The whole dataset is normalized to the first module size, then to each next size, and one combined summary plot is built.">?</span>
              </label>
              <input id="x_targets" name="x_targets" type="text" placeholder="2,3,4,6,8,10" value="{{ x_targets_value or '' }}" {% if run_mode != 'sweep' %}disabled{% endif %}>
              <p class="hint">Comma-separated integer target module sizes (1..200).</p>
            </div>

            <div class="field">
              <label for="engine">Engine</label>
              <select id="engine" name="engine" required>
                {% for engine in engines %}
                <option value="{{ engine }}" {% if selected_engine == engine %}selected{% endif %}>{{ engine }}</option>
                {% endfor %}
              </select>
            </div>
            <div class="field">
              <label for="iterations">Iterations
                <span class="tip" tabindex="0" data-tip="The number of decodings per image. Time is measured for each decoding, and the minimum time is taken. This allows to minimize the influence of third-party processes on the result of the experiment.">?</span>
              </label>
              <input id="iterations" name="iterations" type="number" min="1" value="{{ iterations_value }}" required>
            </div>
            <div class="field">
              <label for="bin_step_px">Module size bin step (px)
                <span class="tip" tabindex="0" data-tip="Bin width for module_size in pixels. Samples are grouped by this step.">?</span>
              </label>
              <input id="bin_step_px" name="bin_step_px" type="number" min="1" value="{{ bin_step_value }}" required>
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
    <script>
      (function () {
        const runForm = document.getElementById("run_form");
        const selectedDisplay = document.getElementById("selected_dataset_display");
        const datasetSourceInput = document.getElementById("dataset_source");
        const datasetNameInput = document.getElementById("dataset_name");
        const fileInput = document.getElementById("dataset_file");
        const browseBtn = document.getElementById("browse_zip_btn");
        const existingBtn = document.getElementById("use_existing_btn");
        const existingPicker = document.getElementById("existing_picker");
        const datasetSelect = document.getElementById("dataset_select");
        const lastDataset = "{{ last_dataset or '' }}";

        const modeInputs = document.querySelectorAll('input[name="run_mode"]');
        const xTargetsField = document.getElementById("x_targets_field");
        const xTargetsInput = document.getElementById("x_targets");

        function setSelectedText(text) {
          selectedDisplay.textContent = text || "No dataset selected";
        }

        function setModeExisting() {
          datasetSourceInput.value = "existing";
          existingPicker.classList.add("open");
          existingPicker.setAttribute("aria-hidden", "false");
          if (datasetSelect && datasetSelect.value) {
            datasetNameInput.value = datasetSelect.value;
            setSelectedText(datasetSelect.value);
          } else if (lastDataset) {
            datasetNameInput.value = lastDataset;
            setSelectedText(lastDataset);
          } else {
            datasetNameInput.value = "";
            setSelectedText("No dataset selected");
          }
          if (datasetSelect) {
            datasetSelect.focus();
          }
        }

        function setModeZip(fileName) {
          datasetSourceInput.value = "zip";
          existingPicker.classList.remove("open");
          existingPicker.setAttribute("aria-hidden", "true");
          datasetNameInput.value = "";
          setSelectedText(fileName || "ZIP selected");
        }

        function activeRunMode() {
          for (const input of modeInputs) {
            if (input.checked) {
              return input.value;
            }
          }
          return "baseline";
        }

        function syncRunModeUi() {
          const isSweep = activeRunMode() === "sweep";
          if (xTargetsField) {
            xTargetsField.classList.toggle("open", isSweep);
          }
          if (xTargetsInput) {
            xTargetsInput.disabled = !isSweep;
            if (!isSweep) {
              xTargetsInput.setCustomValidity("");
            }
          }
        }

        function parseTargets(raw) {
          const tokens = String(raw || "").split(",").map((t) => t.trim()).filter(Boolean);
          if (!tokens.length) {
            return { ok: false, error: "Enter at least one x_target value." };
          }
          const values = [];
          for (const token of tokens) {
            if (!/^\\d+$/.test(token)) {
              return { ok: false, error: "x_targets must contain integers only." };
            }
            const value = Number(token);
            if (!Number.isInteger(value) || value < 1 || value > 200) {
              return { ok: false, error: "x_targets values must be in range 1..200." };
            }
            values.push(value);
          }
          if (!values.length) {
            return { ok: false, error: "Enter at least one x_target value." };
          }
          const dedup = Array.from(new Set(values)).sort((a, b) => a - b);
          return { ok: true, values: dedup };
        }

        if (browseBtn) {
          browseBtn.addEventListener("click", () => {
            fileInput.click();
          });
        }

        if (fileInput) {
          fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length) {
              setModeZip(fileInput.files[0].name);
            }
          });
        }

        if (existingBtn) {
          existingBtn.addEventListener("click", () => {
            setModeExisting();
          });
        }

        if (datasetSelect) {
          datasetSelect.addEventListener("change", () => {
            if (datasetSelect.value) {
              datasetNameInput.value = datasetSelect.value;
              setSelectedText(datasetSelect.value);
              existingPicker.classList.remove("open");
              existingPicker.setAttribute("aria-hidden", "true");
            }
          });
        }

        if (modeInputs.length) {
          modeInputs.forEach((input) => {
            input.addEventListener("change", syncRunModeUi);
          });
        }

        if (runForm) {
          runForm.addEventListener("submit", (event) => {
            if (activeRunMode() !== "sweep" || !xTargetsInput) {
              return;
            }
            const parsed = parseTargets(xTargetsInput.value);
            if (!parsed.ok) {
              event.preventDefault();
              xTargetsInput.setCustomValidity(parsed.error);
              xTargetsInput.reportValidity();
              return;
            }
            xTargetsInput.setCustomValidity("");
            xTargetsInput.value = parsed.values.join(",");
          });
        }

        const structureBtn = document.getElementById("datasetStructureBtn");
        const structureModal = document.getElementById("datasetStructureModal");
        const structureClose = document.getElementById("datasetStructureClose");

        function openStructure() {
          if (!structureModal) return;
          structureModal.classList.add("open");
          structureModal.setAttribute("aria-hidden", "false");
        }

        function closeStructure() {
          if (!structureModal) return;
          structureModal.classList.remove("open");
          structureModal.setAttribute("aria-hidden", "true");
        }

        if (structureBtn) {
          structureBtn.addEventListener("click", openStructure);
        }
        if (structureClose) {
          structureClose.addEventListener("click", closeStructure);
        }
        if (structureModal) {
          structureModal.addEventListener("click", (event) => {
            if (event.target === structureModal) {
              closeStructure();
            }
          });
        }
        document.addEventListener("keydown", (event) => {
          if (event.key === "Escape" && structureModal && structureModal.classList.contains("open")) {
            closeStructure();
          }
        });

        syncRunModeUi();

        if (lastDataset) {
          setSelectedText(lastDataset);
          datasetNameInput.value = lastDataset;
        } else {
          setSelectedText("No dataset selected");
        }
      })();
    </script>

    <div id="datasetStructureModal" class="info-modal" role="dialog" aria-modal="true" aria-labelledby="datasetStructureTitle" aria-hidden="true">
      <div class="info-modal__content">
        <div class="info-modal__header">
          <div id="datasetStructureTitle" class="info-modal__title">Dataset structure</div>
          <button id="datasetStructureClose" class="info-modal__close" type="button" aria-label="Close" onclick="(function(){var m=document.getElementById('datasetStructureModal'); if(m){m.classList.remove('open'); m.setAttribute('aria-hidden','true');}})()">×</button>
        </div>
        <div class="info-modal__body">
          The dataset ZIP (or folder) must contain images and markup in the following layout:
          <pre><code>datasets/&lt;dataset_name&gt;/
  images/QR_CODE/&lt;image&gt;.jpg|png|jpeg
  markup/QR_CODE/&lt;image&gt;.json</code></pre>
          <p>Each markup JSON must include:</p>
          <pre><code>props.barcode.module_size
props.barcode.value</code></pre>
        </div>
      </div>
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
      .progress-note {
        margin-top: 10px;
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
          <span class="meta__item">Mode: <strong>{{ mode_label }}</strong></span>
          {% if x_targets_text %}
          <span class="meta__item">module sizes: <strong>{{ x_targets_text }}</strong>
            <span class="tip" tabindex="0" data-tip="Module sizes are target QR module sizes in pixels. The whole dataset is normalized to the first module size, then to each next size, and one combined summary plot is built.">?</span>
          </span>
          {% endif %}
        </div>
        <div class="progress">
          <div class="progress__fill" id="progress-fill"></div>
        </div>
        <div class="progress__meta">
          <span id="progress-text">0%</span>
          <span id="progress-count">0 / {{ total }}</span>
        </div>
        <div id="progress-note" class="progress-note"></div>
        <div class="summary">
          <div>processed: <span id="processed">0</span></div>
          <div>skipped_no_markup: <span id="skipped_no_markup">0</span>
            <span class="tip" tabindex="0" data-tip="Images skipped because markup JSON is missing or unreadable.">?</span>
          </div>
          <div id="target_progress_box" style="display:none;">target: <span id="target_progress">0 / 0</span></div>
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
      const targetProgressBox = document.getElementById("target_progress_box");
      const targetProgress = document.getElementById("target_progress");
      const progressNote = document.getElementById("progress-note");
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

          if (data.target_total && data.target_total > 0) {
            targetProgressBox.style.display = "block";
            targetProgress.textContent = `${data.target_index || 0} / ${data.target_total}`;
          } else {
            targetProgressBox.style.display = "none";
          }

          if (progressNote) {
            progressNote.textContent = data.progress_note || "";
          }

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
      .explain {
        margin-top: 12px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid rgba(30, 40, 60, 0.12);
        background: #f8fbff;
        color: #334155;
        font-size: 0.88rem;
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
    
      .modal {
        position: fixed;
        inset: 0;
        background: rgba(15, 23, 42, 0.45);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 999;
      }
      .modal.open {
        display: flex;
      }
      .modal__content {
        width: min(900px, 92vw);
        max-height: 85vh;
        background: #ffffff;
        border-radius: 14px;
        box-shadow: 0 24px 80px rgba(15, 23, 42, 0.25);
        padding: 18px 20px 20px;
        overflow: auto;
      }
      .modal__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
      }
      .modal__title {
        font-size: 18px;
        font-weight: 600;
        color: #0f172a;
      }
      .modal__close {
        border: none;
        background: #e2e8f0;
        color: #0f172a;
        border-radius: 999px;
        width: 32px;
        height: 32px;
        cursor: pointer;
        font-size: 18px;
        line-height: 1;
      }
      .modal__status {
        font-size: 14px;
        color: #475569;
        margin-bottom: 10px;
      }
      .modal__list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 10px;
      }
      .modal__item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 12px;
      }
      .modal__item a {
        color: #0f766e;
        text-decoration: none;
        font-weight: 500;
      }
      .modal__item a:hover {
        text-decoration: underline;
      }

      .modal__details {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 0.85rem;
        color: #475569;
      }
      .modal__meta {
        font-size: 0.8rem;
        color: #64748b;
      }
      .modal__action {
        display: flex;
        flex-direction: column;
        gap: 6px;
        align-items: flex-end;
      }
      .modal__button {
        border: none;
        background: #0f8b8d;
        color: #ffffff;
        padding: 6px 10px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 0.85rem;
      }
      .modal__button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .modal__result {
        font-size: 0.8rem;
        color: #334155;
        text-align: right;
        white-space: pre-line;
      }
      .modal__label {
        color: #0f766e;
        font-weight: 600;
      }
    </style>
  </head>
  <body>
    <div class="page">
      <a class="back-link" href="{{ url_for('index', dataset=dataset) }}">Back to setup</a>
      <div class="card">
        <h1>{{ panel_title }}</h1>
        <div class="meta">
          <span class="meta__item">Dataset: <strong>{{ dataset }}</strong></span>
          <span class="meta__item">Engine: <strong>{{ engine }}</strong></span>
          <span class="meta__item">Mode: <strong>{{ mode_label }}</strong></span>
          <span class="meta__item">Iterations: <strong>{{ iterations }}</strong></span>
          <span class="meta__item">Time mode: <strong>{{ time_mode }}</strong></span>
          {% if job_type == 'sweep' and x_targets_text %}
          <span class="meta__item">module sizes: <strong>{{ x_targets_text }}</strong>
            <span class="tip" tabindex="0" data-tip="Module sizes are target QR module sizes in pixels. The whole dataset is normalized to the first module size, then to each next size, and one combined summary plot is built.">?</span>
          </span>
          {% endif %}
        </div>
        <div class="summary">
          <div>processed: {{ processed }}</div>
          <div>skipped_no_markup: {{ skipped_no_markup }}
            <span class="tip" tabindex="0" data-tip="Images skipped because markup JSON is missing or unreadable.">?</span>
          </div>
          <div>GT samples: {{ gt_samples }}
            <span class="tip" tabindex="0" data-tip="Images that have a non-empty expected value (Ground Truth) in markup and can be compared against decoded output.">?</span>
          </div>
          <div>NO GT samples: {{ no_gt_samples }}</div>
        </div>
        <div class="links">
          <a href="{{ plot_url }}" target="_blank" rel="noopener">Open plot</a>
          <a href="{{ html_url }}" target="_blank" rel="noopener">Download HTML</a>
          <a href="{{ json_url }}" target="_blank" rel="noopener">Download JSON</a>
        </div>
        {% if job_type == 'sweep' %}
        <div class="explain">P10-P90 band is the percentile range from the 10th to the 90th percentile of decode time; it shows the typical spread without extreme outliers.</div>
        {% endif %}
      </div>
      <div class="card">
        <iframe src="{{ plot_url_embed }}"></iframe>
      </div>
    </div>
  
    {% if job_type == 'baseline' %}
    <div id="drilldownModal" class="modal">
      <div class="modal__content" role="dialog" aria-modal="true" aria-labelledby="drilldownTitle">
        <div class="modal__header">
          <div id="drilldownTitle" class="modal__title">Samples</div>
          <button id="drilldownClose" class="modal__close" type="button">×</button>
        </div>
        <div id="drilldownStatus" class="modal__status"></div>
        <ul id="drilldownList" class="modal__list"></ul>
      </div>
    </div>

    <script>
      (function () {
        if (window.__drilldownInjected) {
          return;
        }
        window.__drilldownInjected = true;

        const jobId = "{{ job_id }}";
        const dataset = "{{ dataset }}";
        const modal = document.getElementById("drilldownModal");
        const list = document.getElementById("drilldownList");
        const status = document.getElementById("drilldownStatus");
        const title = document.getElementById("drilldownTitle");
        const closeBtn = document.getElementById("drilldownClose");

        function openModal() {
          modal.classList.add("open");
        }

        function closeModal() {
          modal.classList.remove("open");
        }

        closeBtn.addEventListener("click", closeModal);
        modal.addEventListener("click", (event) => {
          if (event.target === modal) {
            closeModal();
          }
        });

        function renderList(items) {
          list.innerHTML = "";
          if (!items.length) {
            status.textContent = "No samples found for this bin/outcome.";
            return;
          }
          status.textContent = `Found ${items.length} samples.`;
          const frag = document.createDocumentFragment();
          for (const item of items) {
            const li = document.createElement("li");
            li.className = "modal__item";

            const details = document.createElement("div");
            details.className = "modal__details";

            const link = document.createElement("a");
            const normalizedPath = normalizeImagePath(item.image_path);
            link.textContent = normalizedPath || item.image_path || "(unknown path)";
            if (normalizedPath) {
              const href = `/api/datasets/${encodeURIComponent(dataset)}/image?path=${encodeURIComponent(normalizedPath)}`;
              link.href = href;
              link.target = "_blank";
              link.rel = "noopener";
            }

            const meta = document.createElement("div");
            meta.className = "modal__meta";
            const time = item.time_total_min_sec != null ? item.time_total_min_sec.toFixed(6) : "n/a";
            meta.textContent = `time=${time}s`;

            details.appendChild(link);
            details.appendChild(meta);

            const action = document.createElement("div");
            action.className = "modal__action";

            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "modal__button";
            btn.textContent = "Run on this image";

            const result = document.createElement("div");
            result.className = "modal__result";

            btn.addEventListener("click", () => runSingle(normalizedPath || item.image_path, btn, result));

            action.appendChild(btn);
            action.appendChild(result);

            li.appendChild(details);
            li.appendChild(action);
            frag.appendChild(li);
          }
          list.appendChild(frag);
        }

        function escapeHtml(value) {
          return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
        }

        function normalizeImagePath(path) {
          if (!path) return "";
          let p = String(path).replace(/\\\\/g, "/");
          p = p.replace(/^\\/+/, "");
          const prefix = `datasets/${dataset}/`;
          if (p.startsWith(prefix)) {
            p = p.slice(prefix.length);
          }
          if (p.startsWith("images/QR_CODE/")) {
            p = p.slice("images/QR_CODE/".length);
          }
          return p;
        }


        async function runSingle(imagePath, button, resultEl) {
          if (!imagePath) {
            resultEl.textContent = "Missing image path.";
            return;
          }
          button.disabled = true;
          resultEl.textContent = "Running...";
          try {
            const resp = await fetch(`/api/jobs/${jobId}/run_single`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ image_path: imagePath }),
            });
            const data = await resp.json();
            if (!resp.ok) {
              resultEl.textContent = data && data.error ? data.error : "Failed.";
              return;
            }
            const time = data.time_total_min_sec != null ? data.time_total_min_sec.toFixed(6) : "n/a";
            const acc = data.accuracy != null ? data.accuracy.toFixed(3) : "n/a";
            const decoded = data.decoded || data.decoded_best || "";
            const expected = data.expected || "";
            resultEl.innerHTML = [
              `<span class="modal__label">time</span> = ${time}s`,
              `<span class="modal__label">accuracy</span> = ${acc}`,
              `<span class="modal__label">decoded</span> = "${escapeHtml(decoded)}"`,
              `<span class="modal__label">expected</span> (if provided) = "${escapeHtml(expected)}"`,
            ].join("<br>");
          } catch (err) {
            resultEl.textContent = "Request failed.";
          } finally {
            button.disabled = false;
          }
        }

        async function fetchSamples(binValue, outcome) {
          title.textContent = `Samples: bin ${binValue} / ${outcome}`;
          status.textContent = "Loading...";
          list.innerHTML = "";
          openModal();
          try {
            const resp = await fetch(`/api/jobs/${jobId}/samples?bin=${encodeURIComponent(binValue)}&outcome=${encodeURIComponent(outcome)}`);
            const raw = await resp.text();
            let data = null;
            try {
              data = JSON.parse(raw);
            } catch (err) {
              data = null;
            }
            if (!resp.ok) {
              status.textContent = data && data.error ? data.error : ("Failed to load samples (" + resp.status + ").");
              return;
            }
            if (!Array.isArray(data)) {
              status.textContent = "Unexpected response format.";
              return;
            }
            renderList(data);
          } catch (err) {
            status.textContent = "Request failed.";
          }
        }

        function attachPlotlyHandler() {
          const iframe = document.querySelector("iframe");
          if (!iframe) {
            return;
          }
          let attached = false;

          const iframeHasDrilldown = () => {
            try {
              const doc = iframe.contentDocument || iframe.contentWindow.document;
              return !!doc && !!doc.getElementById("drilldownModal");
            } catch (err) {
              return false;
            }
          };

          const disableParentModal = () => {
            if (iframeHasDrilldown()) {
              if (modal && modal.parentNode) {
                modal.parentNode.removeChild(modal);
              }
              return true;
            }
            return false;
          };

          const tryAttach = () => {
            if (attached) {
              return true;
            }
            if (disableParentModal()) {
              return true;
            }
            try {
              const doc = iframe.contentDocument || iframe.contentWindow.document;
              const plot = doc.querySelector(".plotly-graph-div");
              if (!plot || !plot.on) {
                return false;
              }
              plot.on("plotly_click", (event) => {
                if (!event || !event.points || !event.points.length) {
                  return;
                }
                const point = event.points[0];
                const traceName = (point.data && point.data.name ? point.data.name : "").toLowerCase();
                let outcome = null;
                if (traceName.includes("failed")) {
                  outcome = "failed";
                } else if (traceName.includes("correct")) {
                  outcome = "correct";
                }
                if (!outcome) {
                  if (traceName.includes("time total min")) {
                    const imagePath = normalizeImagePath(point.customdata);
                    if (!imagePath) {
                      return;
                    }
                    title.textContent = `Sample: ${imagePath}`;
                    renderList([{ image_path: imagePath, time_total_min_sec: point.y }]);
                    openModal();
                  }
                  return;
                }
                const binValue = parseInt(point.x, 10);
                if (!Number.isFinite(binValue)) {
                  return;
                }
                fetchSamples(binValue, outcome);
              });
              attached = true;
              return true;
            } catch (err) {
              return false;
            }
          };

          const scheduleAttach = () => {
            let tries = 0;
            const timer = setInterval(() => {
              tries += 1;
              if (tryAttach() || tries > 30) {
                clearInterval(timer);
              }
            }, 200);
          };

          iframe.addEventListener("load", scheduleAttach);
          scheduleAttach();
        }

        attachPlotlyHandler();

      })();
    </script>
    {% endif %}

  </body>
</html>
"""


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _is_project_root(path: Path) -> bool:
    return (path / "src").is_dir()


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if _is_project_root(cwd):
        _ensure_project_dirs(cwd)
        return cwd

    fallback = Path(__file__).resolve().parents[2]
    if _is_project_root(fallback):
        _ensure_project_dirs(fallback)
        return fallback

    raise RuntimeError("Cannot locate project root. Expected src/.")


def _ensure_project_dirs(project_root: Path) -> None:
    (project_root / "datasets").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)


def _parse_positive_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _parse_run_mode(value: str | None) -> str:
    if (value or "").strip().lower() == "sweep":
        return "sweep"
    return "baseline"


def _parse_x_targets(value: str | None) -> tuple[list[int] | None, str | None]:
    if value is None or not value.strip():
        return None, "x_targets is required for sweep mode."

    targets: list[int] = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if not token.isdigit():
            return None, f"Invalid x_target '{token}'. Use comma-separated integers."
        parsed = int(token)
        if parsed < 1 or parsed > 200:
            return None, "x_targets values must be in range 1..200."
        targets.append(parsed)

    if not targets:
        return None, "x_targets is required for sweep mode."

    return sorted(set(targets)), None


def _safe_engine_key(engine_key: str, registry: dict[str, Any]) -> bool:
    return engine_key in registry


def _list_datasets(project_root: Path) -> list[str]:
    datasets_dir = project_root / "datasets"
    if not datasets_dir.is_dir():
        return []
    names: list[str] = []
    for item in datasets_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name == "_uploads":
            continue
        if _has_dataset_structure(item):
            names.append(item.name)
    return sorted(names)


def _dataset_image_count(project_root: Path, dataset_name: str) -> int:
    images_dir = project_root / "datasets" / dataset_name / "images" / "QR_CODE"
    if not images_dir.is_dir():
        return 0
    count = 0
    for item in images_dir.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            count += 1
    return count


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
    *,
    job_type: str = "baseline",
    x_targets: list[int] | None = None,
    time_mode: str = "time_total_min",
) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "job_type": job_type,
            "dataset": dataset,
            "engine": engine_key,
            "iterations": iterations,
            "bin_step_px": bin_step_px,
            "x_targets": list(x_targets or []),
            "time_mode": time_mode,
            "total": total,
            "seen": 0,
            "processed": 0,
            "skipped_no_markup": 0,
            "target_index": 0,
            "target_total": len(x_targets or []),
            "progress_note": None,
            "heartbeat_ts": time.time(),
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
    _update_job(
        job_id,
        status="running",
        progress_note="Running baseline experiment...",
        heartbeat_ts=time.time(),
    )

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
            heartbeat_ts=time.time(),
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
        seen=summary.processed + summary.skipped_no_markup + summary.skipped_bad_module + summary.skipped_image_read,
        processed=summary.processed,
        skipped_no_markup=summary.skipped_no_markup,
        json_file=out_json.name,
        html_file=out_html.name,
        summary={
            "processed": summary.processed,
            "skipped_no_markup": summary.skipped_no_markup,
        },
        progress_note="Baseline experiment completed.",
        heartbeat_ts=time.time(),
    )


def _run_sweep_job(
    job_id: str,
    project_root: Path,
    dataset: str,
    engine_key: str,
    iterations: int,
    bin_step_px: int,
    x_targets: list[int],
    total_per_target: int,
    time_mode: str,
) -> None:
    target_count = max(1, len(x_targets))
    total_expected = max(0, total_per_target) * target_count

    _update_job(
        job_id,
        status="running",
        progress_note="Starting dataset with a single module size run...",
        heartbeat_ts=time.time(),
    )

    try:
        engine = ENGINE_REGISTRY[engine_key]()
    except Exception as exc:
        _update_job(job_id, status="error", error=f"Failed to initialize engine: {exc}")
        return

    cfg = ExperimentConfig(
        dataset_name=dataset,
        decode_iterations=iterations,
        module_bin_step_px=bin_step_px,
        time_mode=time_mode,
    )

    current_target_idx = 0
    last_seen = 0
    last_processed = 0
    last_skipped_no_markup = 0
    processed_offset = 0
    skipped_no_markup_offset = 0

    def _progress_cb(state: ProgressState) -> None:
        nonlocal current_target_idx
        nonlocal last_seen
        nonlocal last_processed
        nonlocal last_skipped_no_markup
        nonlocal processed_offset
        nonlocal skipped_no_markup_offset

        if state.seen < last_seen:
            processed_offset += last_processed
            skipped_no_markup_offset += last_skipped_no_markup
            current_target_idx = min(current_target_idx + 1, max(0, target_count - 1))

        last_seen = state.seen
        last_processed = state.processed
        last_skipped_no_markup = state.skipped_no_markup

        combined_seen = current_target_idx * total_per_target + state.seen
        combined_processed = processed_offset + state.processed
        combined_skip_markup = skipped_no_markup_offset + state.skipped_no_markup

        target_label = x_targets[current_target_idx] if x_targets else "?"
        _update_job(
            job_id,
            seen=min(total_expected, combined_seen),
            processed=combined_processed,
            skipped_no_markup=combined_skip_markup,
            target_index=current_target_idx + 1,
            target_total=target_count,
            progress_note=f"x_target {current_target_idx + 1}/{target_count} ({target_label} px)",
            heartbeat_ts=time.time(),
        )

    try:
        sweep_results, sweep_summary = run_module_size_normalization_sweep(
            project_root,
            engine,
            cfg,
            x_targets=x_targets,
            progress_cb=_progress_cb,
        )
    except Exception as exc:
        _update_job(job_id, status="error", error=f"Normalization sweep failed: {exc}")
        return

    if not sweep_results:
        _update_job(job_id, status="error", error="No sweep results produced.")
        return

    out_json = save_normalization_sweep_json(
        project_root,
        engine.name,
        dataset,
        x_targets=x_targets,
        results=sweep_results,
        summary=sweep_summary,
    )

    out_html = out_json.with_name(f"qr_normalization_sweep_{dataset}_{engine.name}.html")
    build_sweep_report(
        sweep_results,
        sweep_summary,
        out_html,
        dataset=dataset,
        engine=engine.name,
        iterations=iterations,
        time_mode=time_mode,
        x_targets=x_targets,
        title=f"QR dataset with a single module size ({dataset}, {engine.name})",
    )

    summary_processed = sum(row.processed for row in sweep_summary.targets)
    summary_skip_no_markup = sum(row.skipped_no_markup for row in sweep_summary.targets)

    _update_job(
        job_id,
        status="done",
        seen=total_expected,
        processed=summary_processed,
        skipped_no_markup=summary_skip_no_markup,
        json_file=out_json.name,
        html_file=out_html.name,
        summary={
            "processed": summary_processed,
            "skipped_no_markup": summary_skip_no_markup,
            "targets": [
                {"x_target": float(row.x_target), "processed": row.processed, "skipped_no_markup": row.skipped_no_markup}
                for row in sweep_summary.targets
            ],
        },
        progress_note="Dataset with a single module size run completed.",
        target_index=target_count,
        target_total=target_count,
        heartbeat_ts=time.time(),
    )


def _resolve_job_json_path(project_root: Path, job: dict[str, Any]) -> Path | None:
    json_name = job.get("json_file")
    engine = job.get("engine")
    if not json_name or not engine:
        return None

    outputs_dir = (project_root / "outputs" / f"{engine}_json_and_graphics").resolve()
    json_path = (outputs_dir / str(json_name)).resolve()

    if not json_path.is_file() or not json_path.is_relative_to(outputs_dir):
        return None
    return json_path


def _load_job_json_payload(project_root: Path, job: dict[str, Any]) -> Any:
    json_path = _resolve_job_json_path(project_root, job)
    if json_path is None:
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_result_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records = payload.get("results")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def _compute_gt_no_gt_counts(payload: Any) -> tuple[int, int]:
    gt_samples = 0
    no_gt_samples = 0
    for rec in _iter_result_records(payload):
        expected = str(rec.get("expected") or "")
        if expected.strip():
            gt_samples += 1
        else:
            no_gt_samples += 1
    return gt_samples, no_gt_samples


def create_app(project_root: Path | None = None) -> Flask:
    root = project_root or _resolve_project_root()
    app = Flask(__name__)
    app.config["PROJECT_ROOT"] = root

    @app.get("/")
    def index() -> str:
        requested_dataset = _safe_dataset_name(request.args.get("dataset"))
        return _render_index(last_dataset=requested_dataset)

    def _render_index(
        *,
        error: str | None = None,
        last_dataset: str | None = None,
        run_mode: str = "baseline",
        x_targets_value: str = "2,3,4,6,8,10",
        selected_engine: str | None = None,
        iterations_value: int = 3,
        bin_step_value: int = 2,
    ) -> str:
        engines = list(ENGINE_REGISTRY.keys())
        local_error = error
        if not engines:
            local_error = local_error or "No engines registered."

        available_datasets = _list_datasets(root)
        if last_dataset:
            dataset_dir = root / "datasets" / last_dataset
            if not _has_dataset_structure(dataset_dir):
                last_dataset = None
        if last_dataset is None and available_datasets:
            last_dataset = available_datasets[0]

        dataset_counts = {name: _dataset_image_count(root, name) for name in available_datasets}
        dataset_counts_json = json.dumps(dataset_counts)

        if not selected_engine and engines:
            selected_engine = engines[0]

        return render_template_string(
            INDEX_TEMPLATE,
            engines=engines,
            error=local_error,
            last_dataset=last_dataset,
            available_datasets=available_datasets,
            dataset_counts_json=dataset_counts_json,
            run_mode=run_mode,
            x_targets_value=x_targets_value,
            selected_engine=selected_engine,
            iterations_value=max(1, int(iterations_value)),
            bin_step_value=max(1, int(bin_step_value)),
        )

    @app.post("/run")
    def run_experiment_route() -> str:
        iterations = _parse_positive_int(request.form.get("iterations"), default=3)
        bin_step_px = _parse_positive_int(request.form.get("bin_step_px"), default=2)
        engine_key = request.form.get("engine", "")
        run_mode = _parse_run_mode(request.form.get("run_mode"))
        x_targets_raw = (request.form.get("x_targets") or "").strip()

        if not _safe_engine_key(engine_key, ENGINE_REGISTRY):
            return _render_index(
                error="Unknown engine selected.",
                run_mode=run_mode,
                x_targets_value=x_targets_raw,
                selected_engine=engine_key,
                iterations_value=iterations,
                bin_step_value=bin_step_px,
            )

        dataset_file = request.files.get("dataset_file")
        dataset = None
        error = None
        dataset_source = request.form.get("dataset_source") or "existing"

        if dataset_source == "zip":
            if dataset_file and getattr(dataset_file, "filename", ""):
                dataset, error = _store_uploaded_dataset(dataset_file, root)
            else:
                error = "No dataset zip provided."
        elif dataset_source == "existing":
            dataset_name = _safe_dataset_name(request.form.get("dataset_name"))
            if dataset_name:
                dataset_dir = root / "datasets" / dataset_name
                if _has_dataset_structure(dataset_dir):
                    dataset = dataset_name
                else:
                    error = f"Dataset '{dataset_name}' is missing or has invalid structure."
            else:
                error = "No dataset selected."
        else:
            error = "Invalid dataset source."

        if error:
            fallback_dataset = dataset or _safe_dataset_name(request.form.get("dataset_name"))
            return _render_index(
                error=error,
                last_dataset=fallback_dataset,
                run_mode=run_mode,
                x_targets_value=x_targets_raw,
                selected_engine=engine_key,
                iterations_value=iterations,
                bin_step_value=bin_step_px,
            )

        if run_mode == "sweep":
            x_targets, x_targets_error = _parse_x_targets(x_targets_raw)
            if x_targets_error:
                return _render_index(
                    error=x_targets_error,
                    last_dataset=dataset,
                    run_mode=run_mode,
                    x_targets_value=x_targets_raw,
                    selected_engine=engine_key,
                    iterations_value=iterations,
                    bin_step_value=bin_step_px,
                )
        else:
            x_targets = []

        total = sum(1 for _ in iter_qr_samples(root, dataset))
        if total == 0:
            return _render_index(
                error="No images found in dataset.",
                last_dataset=dataset,
                run_mode=run_mode,
                x_targets_value=x_targets_raw,
                selected_engine=engine_key,
                iterations_value=iterations,
                bin_step_value=bin_step_px,
            )

        job_id = uuid.uuid4().hex
        time_mode = "time_total_min"

        if run_mode == "sweep":
            assert x_targets
            _init_job(
                job_id,
                dataset,
                engine_key,
                iterations,
                bin_step_px,
                total * len(x_targets),
                job_type="sweep",
                x_targets=x_targets,
                time_mode=time_mode,
            )
            thread = threading.Thread(
                target=_run_sweep_job,
                args=(job_id, root, dataset, engine_key, iterations, bin_step_px, x_targets, total, time_mode),
                daemon=True,
            )
            thread.start()
            return redirect(url_for("job_status", job_id=job_id))

        _init_job(
            job_id,
            dataset,
            engine_key,
            iterations,
            bin_step_px,
            total,
            job_type="baseline",
            time_mode=time_mode,
        )

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

        mode_label = "dataset with a single module size" if job.get("job_type") == "sweep" else "Baseline"
        x_targets_text = ", ".join(str(x) for x in job.get("x_targets", []))

        return render_template_string(
            PROGRESS_TEMPLATE,
            job_id=job_id,
            dataset=job["dataset"],
            engine=job["engine"],
            total=job["total"],
            mode_label=mode_label,
            x_targets_text=x_targets_text,
        )

    @app.get("/progress/<job_id>")
    def progress(job_id: str):
        job = _get_job(job_id)
        if job is None:
            abort(404)
        return jsonify(
            {
                "status": job["status"],
                "job_type": job.get("job_type", "baseline"),
                "seen": job["seen"],
                "processed": job["processed"],
                "skipped_no_markup": job["skipped_no_markup"],
                "total": job["total"],
                "target_index": job.get("target_index", 0),
                "target_total": job.get("target_total", 0),
                "progress_note": job.get("progress_note"),
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

        plot_url = url_for("serve_output", engine=job["engine"], filename=job["html_file"], job_id=job_id)
        plot_url_embed = f"{plot_url}&embed=1"
        json_url = url_for("serve_output", engine=job["engine"], filename=job["json_file"])
        html_url = url_for("serve_output", engine=job["engine"], filename=job["html_file"])

        job_type = job.get("job_type", "baseline")
        mode_label = "dataset with a single module size" if job_type == "sweep" else "Baseline"
        x_targets_text = ", ".join(str(x) for x in job.get("x_targets", []))

        payload = _load_job_json_payload(root, job)
        gt_samples, no_gt_samples = _compute_gt_no_gt_counts(payload)

        summary = job.get("summary") or {}
        processed = int(summary.get("processed", job.get("processed", 0) or 0))
        skipped_no_markup = int(summary.get("skipped_no_markup", job.get("skipped_no_markup", 0) or 0))

        if job_type == "sweep":
            panel_title = f"QR dataset with a single module size ({job['dataset']}, {job['engine']})"
        else:
            panel_title = f"QR baseline run ({job['dataset']}, {job['engine']})"

        return render_template_string(
            RESULT_TEMPLATE,
            job_id=job_id,
            job_type=job_type,
            panel_title=panel_title,
            mode_label=mode_label,
            dataset=job["dataset"],
            engine=job["engine"],
            iterations=job["iterations"],
            time_mode=job.get("time_mode", "time_total_min"),
            x_targets_text=x_targets_text,
            processed=processed,
            skipped_no_markup=skipped_no_markup,
            gt_samples=gt_samples,
            no_gt_samples=no_gt_samples,
            plot_url=plot_url,
            plot_url_embed=plot_url_embed,
            json_url=json_url,
            html_url=html_url,
        )

    @app.get("/api/jobs/<job_id>/samples")
    def job_samples(job_id: str):
        job = _get_job(job_id)
        if job is None:
            return jsonify({"error": "Unknown job_id."}), 404

        if job.get("status") != "done" or not job.get("json_file"):
            return jsonify({"error": "Results not ready for this job."}), 400

        payload = _load_job_json_payload(root, job)
        if payload is None:
            return jsonify({"error": "Results JSON not found."}), 404

        job_type = job.get("job_type", "baseline")

        if job_type == "sweep":
            if not isinstance(payload, dict):
                return jsonify({"error": "Sweep results JSON has invalid format."}), 500

            records = payload.get("results")
            if not isinstance(records, list):
                return jsonify({"error": "Sweep results JSON has invalid format."}), 500

            x_target_param = request.args.get("x_target")
            gt_only_flag = (request.args.get("gt_only") or "").strip().lower() in {"1", "true", "yes"}
            no_gt_only_flag = (request.args.get("no_gt_only") or "").strip().lower() in {"1", "true", "yes"}

            if gt_only_flag and no_gt_only_flag:
                return jsonify({"error": "gt_only and no_gt_only cannot be used together."}), 400

            x_target_filter: float | None = None
            if x_target_param not in {None, ""}:
                try:
                    x_target_filter = float(x_target_param)
                except ValueError:
                    return jsonify({"error": "Invalid x_target value."}), 400

            try:
                limit = int(request.args.get("limit") or 100)
                offset = int(request.args.get("offset") or 0)
            except ValueError:
                return jsonify({"error": "limit and offset must be integers."}), 400

            if limit < 1 or limit > 500:
                return jsonify({"error": "limit must be in range 1..500."}), 400
            if offset < 0:
                return jsonify({"error": "offset must be >= 0."}), 400

            filtered: list[dict[str, Any]] = []
            for sample_index, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue

                if x_target_filter is not None:
                    try:
                        rec_target = float(rec.get("x_target"))
                    except (TypeError, ValueError):
                        continue
                    if abs(rec_target - x_target_filter) > 1e-9:
                        continue

                expected = str(rec.get("expected") or "")
                no_gt = not expected.strip()

                if gt_only_flag and no_gt:
                    continue
                if no_gt_only_flag and not no_gt:
                    continue

                accuracy_value: float | None
                if no_gt:
                    accuracy_value = None
                else:
                    try:
                        accuracy_value = float(rec.get("accuracy"))
                    except (TypeError, ValueError):
                        accuracy_value = None

                filtered.append(
                    {
                        "sample_index": sample_index,
                        "image_path": rec.get("image_path"),
                        "x_target": rec.get("x_target"),
                        "module_size_px_current": rec.get("module_size_px_current"),
                        "scale_factor": rec.get("scale_factor"),
                        "resized_width": rec.get("resized_width"),
                        "resized_height": rec.get("resized_height"),
                        "decode_time_ms": rec.get("decode_time_ms"),
                        "decode_success": bool(rec.get("decode_success")),
                        "decoded": rec.get("decoded"),
                        "expected": expected,
                        "no_gt": no_gt,
                        "accuracy": accuracy_value,
                    }
                )

            return jsonify(filtered[offset : offset + limit])

        if not isinstance(payload, list):
            return jsonify({"error": "Results JSON has invalid format."}), 500

        bin_param = request.args.get("bin")
        outcome = (request.args.get("outcome") or "").strip().lower()

        if bin_param is None:
            return jsonify({"error": "Missing required query param: bin."}), 400
        try:
            bin_value = int(bin_param)
        except ValueError:
            return jsonify({"error": "Invalid bin value. Must be an integer."}), 400
        if bin_value <= 0:
            return jsonify({"error": "Invalid bin value. Must be > 0."}), 400

        if outcome not in {"failed", "correct"}:
            return jsonify({"error": "Invalid outcome. Use 'failed' or 'correct'."}), 400

        dataset_prefix = f"datasets/{job['dataset']}/"

        def _time_value(rec: dict) -> float | None:
            for key in ("time_total_min_sec", "time_total_min", "time"):
                if key in rec and rec[key] is not None:
                    try:
                        return float(rec[key])
                    except (TypeError, ValueError):
                        return None
            return None

        def _accuracy_match(rec: dict) -> bool:
            try:
                acc = float(rec.get("accuracy"))
            except (TypeError, ValueError):
                return False
            if outcome == "failed":
                return acc == 0.0
            return acc == 1.0

        filtered = []
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            try:
                module_size = int(rec.get("module_size"))
            except (TypeError, ValueError):
                continue
            if module_size != bin_value:
                continue
            if not _accuracy_match(rec):
                continue

            raw_path = str(rec.get("image_path") or "")
            clean_path = raw_path
            if clean_path.startswith(dataset_prefix):
                clean_path = clean_path[len(dataset_prefix):]
            elif clean_path.startswith("/" + dataset_prefix):
                clean_path = clean_path[len(dataset_prefix) + 1 :]

            if not clean_path.startswith("images/QR_CODE/"):
                continue
            if ".." in Path(clean_path).parts:
                continue

            filtered.append(
                {
                    "image_path": clean_path,
                    "module_size_raw": rec.get("module_size_raw"),
                    "time_total_min_sec": _time_value(rec),
                    "decoded": rec.get("decoded"),
                    "expected": rec.get("expected"),
                }
            )

        return jsonify(filtered)

    @app.get("/api/jobs/<job_id>/sample/<sample_index>")
    def job_sample_details(job_id: str, sample_index: str):
        job = _get_job(job_id)
        if job is None:
            return jsonify({"error": "Unknown job_id."}), 404

        if job.get("status") != "done" or not job.get("json_file"):
            return jsonify({"error": "Results not ready for this job."}), 400

        try:
            index = int(sample_index)
        except ValueError:
            return jsonify({"error": "sample_index must be an integer."}), 400
        if index < 0:
            return jsonify({"error": "sample_index must be >= 0."}), 400

        payload = _load_job_json_payload(root, job)
        if payload is None:
            return jsonify({"error": "Results JSON not found."}), 404

        job_type = job.get("job_type", "baseline")

        if job_type == "sweep":
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                return jsonify({"error": "Sweep results JSON has invalid format."}), 500
            records = payload["results"]
            if index >= len(records):
                return jsonify({"error": "sample_index out of range."}), 404

            rec = records[index]
            if not isinstance(rec, dict):
                return jsonify({"error": "Sample record has invalid format."}), 500

            expected = str(rec.get("expected") or "")
            no_gt = not expected.strip()
            accuracy_value: float | None = None
            if not no_gt:
                try:
                    accuracy_value = float(rec.get("accuracy"))
                except (TypeError, ValueError):
                    accuracy_value = None

            return jsonify(
                {
                    "sample_index": index,
                    "dataset": job.get("dataset"),
                    "image_path": rec.get("image_path"),
                    "x_target": rec.get("x_target"),
                    "module_size_px_current": rec.get("module_size_px_current"),
                    "scale_factor": rec.get("scale_factor"),
                    "resized_width": rec.get("resized_width"),
                    "resized_height": rec.get("resized_height"),
                    "decode_time_ms": rec.get("decode_time_ms"),
                    "decode_success": bool(rec.get("decode_success")),
                    "decoded": rec.get("decoded"),
                    "expected": expected,
                    "no_gt": no_gt,
                    "accuracy": accuracy_value,
                    "note": "Ground-truth is not provided for this image. Accuracy comparison is skipped." if no_gt else "",
                }
            )

        if not isinstance(payload, list):
            return jsonify({"error": "Results JSON has invalid format."}), 500
        if index >= len(payload):
            return jsonify({"error": "sample_index out of range."}), 404

        rec = payload[index]
        if not isinstance(rec, dict):
            return jsonify({"error": "Sample record has invalid format."}), 500

        expected = str(rec.get("expected") or "")
        no_gt = not expected.strip()

        time_value: float | None = None
        for key in ("time_total_min_sec", "time_total_min", "time"):
            if key in rec and rec[key] is not None:
                try:
                    time_value = float(rec[key])
                except (TypeError, ValueError):
                    time_value = None
                break

        return jsonify(
            {
                "sample_index": index,
                "dataset": job.get("dataset"),
                "image_path": rec.get("image_path"),
                "module_size": rec.get("module_size"),
                "module_size_raw": rec.get("module_size_raw"),
                "time_total_min_sec": time_value,
                "decoded": rec.get("decoded"),
                "expected": expected,
                "no_gt": no_gt,
                "accuracy": rec.get("accuracy"),
            }
        )

    @app.get("/api/datasets/<dataset>/image")
    def dataset_image(dataset: str):
        safe_dataset = _safe_dataset_name(dataset)
        if safe_dataset is None or safe_dataset != dataset:
            return jsonify({"error": "Invalid dataset name."}), 400

        rel_path = request.args.get("path")
        if not rel_path:
            return jsonify({"error": "Missing required query param: path."}), 400

        normalized = rel_path.replace("\\", "/").lstrip("/")
        dataset_prefix = f"datasets/{safe_dataset}/"
        if normalized.startswith(dataset_prefix):
            normalized = normalized[len(dataset_prefix):]
        prefix = "images/QR_CODE/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

        rel = Path(normalized)
        if rel.is_absolute() or ".." in rel.parts:
            return jsonify({"error": "Invalid path."}), 400

        images_dir = (root / "datasets" / safe_dataset / "images" / "QR_CODE").resolve()
        if not images_dir.is_dir():
            return jsonify({"error": "Dataset images folder not found."}), 404

        file_path = (images_dir / rel).resolve()
        if not file_path.is_file() or not file_path.is_relative_to(images_dir):
            return jsonify({"error": "Image not found."}), 404

        suffix = file_path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            return jsonify({"error": "Unsupported image type."}), 400

        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        return file_path.read_bytes(), 200, {"Content-Type": content_type}

    @app.post("/api/jobs/<job_id>/run_single")
    def run_single(job_id: str):
        job = _get_job(job_id)
        if job is None:
            return jsonify({"error": "Unknown job_id."}), 404

        payload = request.get_json(silent=True) or request.form or {}
        image_path = payload.get("image_path") if isinstance(payload, dict) else None
        if not image_path or not isinstance(image_path, str):
            return jsonify({"error": "Missing or invalid image_path."}), 400

        dataset = job.get("dataset")
        engine_key = job.get("engine")
        iterations = int(job.get("iterations") or 1)

        if not dataset or not engine_key:
            return jsonify({"error": "Job metadata incomplete."}), 400
        if not _safe_engine_key(engine_key, ENGINE_REGISTRY):
            return jsonify({"error": "Unknown engine."}), 400

        safe_dataset = _safe_dataset_name(dataset)
        if safe_dataset is None or safe_dataset != dataset:
            return jsonify({"error": "Invalid dataset."}), 400

        dataset_root = (root / "datasets" / safe_dataset).resolve()
        images_dir = (dataset_root / "images" / "QR_CODE").resolve()
        if not images_dir.is_dir():
            return jsonify({"error": "Dataset images folder not found."}), 404

        normalized_path = image_path.replace("\\", "/").lstrip("/")
        dataset_prefix = f"datasets/{safe_dataset}/"
        if normalized_path.startswith(dataset_prefix):
            normalized_path = normalized_path[len(dataset_prefix):]

        image_prefix = "images/QR_CODE/"
        rel_candidate = normalized_path
        if normalized_path.startswith(image_prefix):
            rel_candidate = normalized_path[len(image_prefix):]

        rel = Path(rel_candidate)
        if rel.is_absolute() or ".." in rel.parts:
            return jsonify({"error": "Invalid image_path."}), 400

        candidate = (images_dir / rel).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(images_dir):
            alt_rel = Path(normalized_path)
            if alt_rel.is_absolute() or ".." in alt_rel.parts:
                return jsonify({"error": "Invalid image_path."}), 400
            candidate = (dataset_root / alt_rel).resolve()
            if not candidate.is_file() or not candidate.is_relative_to(images_dir):
                return jsonify({"error": "Image not found or outside images/QR_CODE."}), 404

        markup_dir = (dataset_root / "markup" / "QR_CODE").resolve()
        markup_path = (markup_dir / f"{candidate.name}.json").resolve()
        if not markup_path.is_file() or not markup_path.is_relative_to(markup_dir):
            return jsonify({"error": "Markup not found for image."}), 404

        markup = read_markup(markup_path)
        if markup is None:
            return jsonify({"error": "Failed to read markup."}), 400
        expected = extract_expected_value(markup)
        if expected is None:
            expected = ""

        try:
            engine = ENGINE_REGISTRY[engine_key]()
        except Exception as exc:
            return jsonify({"error": f"Failed to initialize engine: {exc}"}), 500

        decoded_values = []
        times = []
        for _ in range(max(1, iterations)):
            try:
                decoded, time_total = engine.decode_once(candidate)
            except EngineError as exc:
                return jsonify({"error": f"Decode failed: {exc}"}), 500
            decoded_text = decoded.decode("utf-8", errors="ignore") if isinstance(decoded, bytes) else str(decoded)
            decoded_values.append(decoded_text)
            times.append(float(time_total))

        if not times:
            return jsonify({"error": "No timing data collected."}), 500

        time_min = min(times)
        decoded_best = Counter(decoded_values).most_common(1)[0][0] if decoded_values else ""
        success_list = [1.0 if val.strip() else 0.0 for val in decoded_values]
        accuracy = sum(success_list) / float(len(success_list)) if success_list else 0.0

        return jsonify(
            {
                "decoded_values": decoded_values,
                "decoded_best": decoded_best,
                "decoded": decoded_best,
                "expected": expected,
                "time_total_min_sec": float(time_min),
                "accuracy": float(accuracy),
            }
        )


    @app.get("/outputs/<engine>/<path:filename>")
    def serve_output(engine: str, filename: str):
        outputs_dir = (root / "outputs" / f"{engine}_json_and_graphics").resolve()
        file_path = (outputs_dir / filename).resolve()

        if not file_path.is_file():
            abort(404)
        if not file_path.is_relative_to(outputs_dir):
            abort(404)

        if file_path.suffix.lower() == ".html":
            html = file_path.read_text(encoding="utf-8")
            job_id = None
            job = None

            job_id_param = request.args.get("job_id")
            if job_id_param:
                candidate = _get_job(job_id_param)
                if candidate and candidate.get("html_file") == filename:
                    job_id = job_id_param
                    job = candidate

            if job is None:
                with JOBS_LOCK:
                    matches = [(jid, j) for jid, j in JOBS.items() if j.get("html_file") == filename]
                if matches:
                    job_id, job = matches[-1]

            if job and job.get("job_type", "baseline") == "baseline" and request.args.get("embed") != "1":
                html = _inject_plot_drilldown(html, job_id, job.get("dataset", ""))
            if job and job.get("job_type", "baseline") == "sweep" and request.args.get("embed") == "1":
                html = _inject_sweep_embed_cleanup(html)
            return html.encode("utf-8"), 200, {
                "Content-Type": "text/html; charset=utf-8",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            }

        return file_path.read_bytes(), 200, {"Content-Type": _content_type(file_path)}

    return app


def _inject_plot_drilldown(html: str, job_id: str, dataset: str) -> str:
    if "</body>" not in html:
        return html

    template = """    <style>
      .modal {
        position: fixed;
        inset: 0;
        background: rgba(15, 23, 42, 0.45);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 999;
      }
      .modal.open {
        display: flex;
      }
      .modal__content {
        width: min(900px, 92vw);
        max-height: 85vh;
        background: #ffffff;
        border-radius: 14px;
        box-shadow: 0 24px 80px rgba(15, 23, 42, 0.25);
        padding: 18px 20px 20px;
        overflow: auto;
      }
      .modal__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
      }
      .modal__title {
        font-size: 18px;
        font-weight: 600;
        color: #0f172a;
      }
      .modal__close {
        border: none;
        background: #e2e8f0;
        color: #0f172a;
        border-radius: 999px;
        width: 32px;
        height: 32px;
        cursor: pointer;
        font-size: 18px;
        line-height: 1;
      }
      .modal__status {
        font-size: 14px;
        color: #475569;
        margin-bottom: 10px;
      }
      .modal__list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 10px;
      }
      .modal__item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 12px;
      }
      .modal__item a {
        color: #0f766e;
        text-decoration: none;
        font-weight: 500;
      }
      .modal__item a:hover {
        text-decoration: underline;
      }
      .modal__details {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 0.85rem;
        color: #475569;
      }
      .modal__meta {
        font-size: 0.8rem;
        color: #64748b;
      }
      .modal__action {
        display: flex;
        flex-direction: column;
        gap: 6px;
        align-items: flex-end;
      }
      .modal__button {
        border: none;
        background: #0f8b8d;
        color: #ffffff;
        padding: 6px 10px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 0.85rem;
      }
      .modal__button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .modal__result {
        font-size: 0.8rem;
        color: #334155;
        text-align: right;
        white-space: pre-line;
      }
      .modal__label {
        color: #0f766e;
        font-weight: 600;
      }
    </style>
    <div id="drilldownModal" class="modal">
      <div class="modal__content" role="dialog" aria-modal="true" aria-labelledby="drilldownTitle">
        <div class="modal__header">
          <div id="drilldownTitle" class="modal__title">Samples</div>
          <button id="drilldownClose" class="modal__close" type="button">×</button>
        </div>
        <div id="drilldownStatus" class="modal__status"></div>
        <ul id="drilldownList" class="modal__list"></ul>
      </div>
    </div>

    <script>
      (function () {
        const jobId = "__JOB_ID__";
        const dataset = "__DATASET__";
        const modal = document.getElementById("drilldownModal");
        const list = document.getElementById("drilldownList");
        const status = document.getElementById("drilldownStatus");
        const title = document.getElementById("drilldownTitle");
        const closeBtn = document.getElementById("drilldownClose");

        function openModal() {
          modal.classList.add("open");
        }

        function closeModal() {
          modal.classList.remove("open");
        }

        closeBtn.addEventListener("click", closeModal);
        modal.addEventListener("click", (event) => {
          if (event.target === modal) {
            closeModal();
          }
        });

        function renderList(items) {
          list.innerHTML = "";
          if (!items.length) {
            status.textContent = "No samples found for this bin/outcome.";
            return;
          }
          status.textContent = `Found ${items.length} samples.`;
          const frag = document.createDocumentFragment();
          for (const item of items) {
            const li = document.createElement("li");
            li.className = "modal__item";

            const details = document.createElement("div");
            details.className = "modal__details";

            const link = document.createElement("a");
            const normalizedPath = normalizeImagePath(item.image_path);
            link.textContent = normalizedPath || item.image_path || "(unknown path)";
            if (normalizedPath) {
              const href = `/api/datasets/${encodeURIComponent(dataset)}/image?path=${encodeURIComponent(normalizedPath)}`;
              link.href = href;
              link.target = "_blank";
              link.rel = "noopener";
            }

            const meta = document.createElement("div");
            meta.className = "modal__meta";
            const time = item.time_total_min_sec != null ? item.time_total_min_sec.toFixed(6) : "n/a";
            meta.textContent = `time=${time}s`;

            details.appendChild(link);
            details.appendChild(meta);

            const action = document.createElement("div");
            action.className = "modal__action";

            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "modal__button";
            btn.textContent = "Run on this image";

            const result = document.createElement("div");
            result.className = "modal__result";

            btn.addEventListener("click", () => runSingle(normalizedPath || item.image_path, btn, result));

            action.appendChild(btn);
            action.appendChild(result);

            li.appendChild(details);
            li.appendChild(action);
            frag.appendChild(li);
          }
          list.appendChild(frag);
        }

        function escapeHtml(value) {
          return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\\"/g, "&quot;")
            .replace(/'/g, "&#39;");
        }

function normalizeImagePath(path) {
          if (!path) return "";
          let p = String(path).replace(/\\\\/g, "/");
          p = p.replace(/^\\/+/, "");
          const prefix = `datasets/${dataset}/`;
          if (p.startsWith(prefix)) {
            p = p.slice(prefix.length);
          }
          if (p.startsWith("images/QR_CODE/")) {
            p = p.slice("images/QR_CODE/".length);
          }
          return p;
        }

        async function runSingle(imagePath, button, resultEl) {
          if (!imagePath) {
            resultEl.textContent = "Missing image path.";
            return;
          }
          button.disabled = true;
          resultEl.textContent = "Running...";
          try {
            const resp = await fetch(`/api/jobs/${jobId}/run_single`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ image_path: imagePath }),
            });
            const data = await resp.json();
            if (!resp.ok) {
              resultEl.textContent = data && data.error ? data.error : "Failed.";
              return;
            }
            const time = data.time_total_min_sec != null ? data.time_total_min_sec.toFixed(6) : "n/a";
            const acc = data.accuracy != null ? data.accuracy.toFixed(3) : "n/a";
                        const decoded = data.decoded || data.decoded_best || "";
            const expected = data.expected || "";
            resultEl.innerHTML = [
              `<span class=\"modal__label\">time</span> = ${time}s`,
              `<span class=\"modal__label\">accuracy</span> = ${acc}`,
              `<span class=\"modal__label\">decoded</span> = \"${escapeHtml(decoded)}\"`,
              `<span class=\"modal__label\">expected</span> (if provided) = \"${escapeHtml(expected)}\"`,
            ].join("<br>");
          } catch (err) {
            resultEl.textContent = "Request failed.";
          } finally {
            button.disabled = false;
          }
        }

        async function fetchSamples(binValue, outcome) {
          title.textContent = `Samples: bin ${binValue} / ${outcome}`;
          status.textContent = "Loading...";
          list.innerHTML = "";
          openModal();
          try {
            const resp = await fetch(`/api/jobs/${jobId}/samples?bin=${encodeURIComponent(binValue)}&outcome=${encodeURIComponent(outcome)}`);
            const raw = await resp.text();
            let data = null;
            try {
              data = JSON.parse(raw);
            } catch (err) {
              data = null;
            }
            if (!resp.ok) {
              status.textContent = data && data.error ? data.error : ("Failed to load samples (" + resp.status + ").");
              return;
            }
            if (!Array.isArray(data)) {
              status.textContent = "Unexpected response format.";
              return;
            }
            renderList(data);
          } catch (err) {
            status.textContent = "Request failed.";
          }
        }

        function attachPlotlyHandler() {
          const plot = document.querySelector(".plotly-graph-div");
          if (!plot || !plot.on) {
            return;
          }
          plot.on("plotly_click", (event) => {
            if (!event || !event.points || !event.points.length) {
              return;
            }
            const point = event.points[0];
            const traceName = (point.data && point.data.name ? point.data.name : "").toLowerCase();
            if (traceName.includes("failed") || traceName.includes("correct")) {
              const outcome = traceName.includes("failed") ? "failed" : "correct";
              const binValue = parseInt(point.x, 10);
              if (!Number.isFinite(binValue)) {
                return;
              }
              fetchSamples(binValue, outcome);
              return;
            }
            if (traceName.includes("time total min")) {
              const imagePath = normalizeImagePath(point.customdata);
              if (!imagePath) {
                return;
              }
              title.textContent = `Sample: ${imagePath}`;
              renderList([{ image_path: imagePath, time_total_min_sec: point.y }]);
              openModal();
            }
          });
        }

        attachPlotlyHandler();
      })();
    </script>"""
    injection = template.replace("__JOB_ID__", job_id).replace("__DATASET__", dataset)
    return html.replace("</body>", injection + "</body>")



def _inject_sweep_embed_cleanup(html: str) -> str:
    if "</head>" not in html:
        return html

    marker = "/* sweep-embed-cleanup */"
    if marker in html:
        return html

    style = (
        "<style>\n"
        "  /* sweep-embed-cleanup */\n"
        "  body .layout > section.card:first-child { display: none !important; }\n"
        "</style>\n"
    )
    return html.replace("</head>", style + "</head>", 1)


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
