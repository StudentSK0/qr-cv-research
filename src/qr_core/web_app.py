from __future__ import annotations

import json
import re
import shutil
import threading
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
from .metrics import ExperimentConfig, ProgressState, run_experiment, save_results_json
from .markup import extract_expected_value, read_markup
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
      .secondary-button:hover
      #browse_zip_btn {
        box-shadow: none;
      }
      #browse_zip_btn:hover,
      #browse_zip_btn:focus {
        box-shadow: 0 16px 28px rgba(15, 139, 141, 0.32);
        transform: translateY(-1px);
      }
 {
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(15, 139, 141, 0.18);
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
        <form method="post" action="{{ url_for('run_experiment_route') }}" enctype="multipart/form-data">

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
    <script>
      (function () {
        const selectedDisplay = document.getElementById("selected_dataset_display");
        const datasetSourceInput = document.getElementById("dataset_source");
        const datasetNameInput = document.getElementById("dataset_name");
        const fileInput = document.getElementById("dataset_file");
        const browseBtn = document.getElementById("browse_zip_btn");
        const existingBtn = document.getElementById("use_existing_btn");
        const existingPicker = document.getElementById("existing_picker");
        const datasetSelect = document.getElementById("dataset_select");
        const lastDataset = "{{ last_dataset or '' }}";


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
        if (lastDataset) {
          setSelectedText(lastDataset);
          datasetNameInput.value = lastDataset;
        } else {
          setSelectedText("No dataset selected");
        }
      })();
    </script>


  
    </div>

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
      }
    </style>
  </head>
  <body>
    <div class="page">
      <div class="card">
        <a class="back-link" href="{{ url_for('index', dataset=dataset) }}">Back to setup</a>
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
            link.textContent = item.image_path || "(unknown path)";
            if (item.image_path) {
              const href = `/api/datasets/${encodeURIComponent(dataset)}/image?path=${encodeURIComponent(item.image_path)}`;
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

            btn.addEventListener("click", () => runSingle(item.image_path, btn, result));

            action.appendChild(btn);
            action.appendChild(result);

            li.appendChild(details);
            li.appendChild(action);
            frag.appendChild(li);
          }
          list.appendChild(frag);
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
            resultEl.textContent = `acc=${acc}, time=${time}s, best=${data.decoded_best || ""}`;
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
            const data = await resp.json();
            if (!resp.ok) {
              status.textContent = data && data.error ? data.error : "Failed to load samples.";
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

        available_datasets = _list_datasets(root)
        last_dataset = _safe_dataset_name(request.args.get("dataset"))
        if last_dataset:
            dataset_dir = root / "datasets" / last_dataset
            if not _has_dataset_structure(dataset_dir):
                last_dataset = None
        if last_dataset is None and available_datasets:
            last_dataset = available_datasets[0]

        dataset_counts = {name: _dataset_image_count(root, name) for name in available_datasets}
        dataset_counts_json = json.dumps(dataset_counts)

        return render_template_string(
            INDEX_TEMPLATE,
            engines=engines,
            error=error,
            last_dataset=last_dataset,
            available_datasets=available_datasets,
            dataset_counts_json=dataset_counts_json,
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
            return render_template_string(
                INDEX_TEMPLATE,
                engines=engines,
                error=error,
                last_dataset=dataset,
                available_datasets=_list_datasets(root),
                dataset_counts_json=json.dumps({name: _dataset_image_count(root, name) for name in _list_datasets(root)}),
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

        plot_url = url_for("serve_output", engine=job["engine"], filename=job["html_file"], job_id=job_id)
        json_url = url_for("serve_output", engine=job["engine"], filename=job["json_file"])

        return render_template_string(
            RESULT_TEMPLATE,
            job_id=job_id,
            dataset=job["dataset"],
            engine=job["engine"],
            iterations=job["iterations"],
            bin_step_px=job["bin_step_px"],
            summary=job["summary"],
            plot_url=plot_url,
            json_url=json_url,
        )

    @app.get("/api/jobs/<job_id>/samples")
    def job_samples(job_id: str):
        job = _get_job(job_id)
        if job is None:
            return jsonify({"error": "Unknown job_id."}), 404

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

        if job.get("status") != "done" or not job.get("json_file"):
            return jsonify({"error": "Results not ready for this job."}), 400

        outputs_dir = (root / "outputs" / f"{job['engine']}_json_and_graphics").resolve()
        json_path = (outputs_dir / job["json_file"]).resolve()

        if not json_path.is_file() or not json_path.is_relative_to(outputs_dir):
            return jsonify({"error": "Results JSON not found."}), 404

        try:
            records = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return jsonify({"error": f"Failed to read results JSON: {exc}"}), 500

        if not isinstance(records, list):
            return jsonify({"error": "Results JSON has invalid format."}), 500

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
        for rec in records:
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

    @app.get("/api/datasets/<dataset>/image")
    def dataset_image(dataset: str):
        safe_dataset = _safe_dataset_name(dataset)
        if safe_dataset is None or safe_dataset != dataset:
            return jsonify({"error": "Invalid dataset name."}), 400

        rel_path = request.args.get("path")
        if not rel_path:
            return jsonify({"error": "Missing required query param: path."}), 400

        normalized = rel_path.replace("\\", "/")
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

        rel = Path(image_path)
        if rel.is_absolute() or ".." in rel.parts:
            return jsonify({"error": "Invalid image_path."}), 400

        candidate = (dataset_root / rel).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(images_dir):
            # Allow passing just filename (relative to images/QR_CODE)
            candidate = (images_dir / rel).resolve()
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
            return jsonify({"error": "Expected value missing in markup."}), 400

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
        accuracy = 1 if decoded_best == expected else 0

        return jsonify(
            {
                "decoded_values": decoded_values,
                "decoded_best": decoded_best,
                "time_total_min_sec": float(time_min),
                "accuracy": int(accuracy),
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
                if len(matches) == 1:
                    job_id, job = matches[0]

            if job:
                html = _inject_plot_drilldown(html, job_id, job.get("dataset", ""))
            return html.encode("utf-8"), 200, {"Content-Type": "text/html; charset=utf-8"}

        return file_path.read_bytes(), 200, {"Content-Type": _content_type(file_path)}

    return app


def _inject_plot_drilldown(html: str, job_id: str, dataset: str) -> str:
    if "</body>" not in html:
        return html

    template = '    <style>\n      .modal {\n        position: fixed;\n        inset: 0;\n        background: rgba(15, 23, 42, 0.45);\n        display: none;\n        align-items: center;\n        justify-content: center;\n        z-index: 999;\n      }\n      .modal.open {\n        display: flex;\n      }\n      .modal__content {\n        width: min(900px, 92vw);\n        max-height: 85vh;\n        background: #ffffff;\n        border-radius: 14px;\n        box-shadow: 0 24px 80px rgba(15, 23, 42, 0.25);\n        padding: 18px 20px 20px;\n        overflow: auto;\n      }\n      .modal__header {\n        display: flex;\n        align-items: center;\n        justify-content: space-between;\n        gap: 12px;\n        margin-bottom: 8px;\n      }\n      .modal__title {\n        font-size: 18px;\n        font-weight: 600;\n        color: #0f172a;\n      }\n      .modal__close {\n        border: none;\n        background: #e2e8f0;\n        color: #0f172a;\n        border-radius: 999px;\n        width: 32px;\n        height: 32px;\n        cursor: pointer;\n        font-size: 18px;\n        line-height: 1;\n      }\n      .modal__status {\n        font-size: 14px;\n        color: #475569;\n        margin-bottom: 10px;\n      }\n      .modal__list {\n        list-style: none;\n        padding: 0;\n        margin: 0;\n        display: grid;\n        gap: 10px;\n      }\n      .modal__item {\n        display: flex;\n        align-items: center;\n        justify-content: space-between;\n        gap: 12px;\n        border: 1px solid #e2e8f0;\n        border-radius: 10px;\n        padding: 10px 12px;\n      }\n      .modal__item a {\n        color: #0f766e;\n        text-decoration: none;\n        font-weight: 500;\n      }\n      .modal__item a:hover {\n        text-decoration: underline;\n      }\n      .modal__details {\n        display: flex;\n        flex-direction: column;\n        gap: 4px;\n        font-size: 0.85rem;\n        color: #475569;\n      }\n      .modal__meta {\n        font-size: 0.8rem;\n        color: #64748b;\n      }\n      .modal__action {\n        display: flex;\n        flex-direction: column;\n        gap: 6px;\n        align-items: flex-end;\n      }\n      .modal__button {\n        border: none;\n        background: #0f8b8d;\n        color: #ffffff;\n        padding: 6px 10px;\n        border-radius: 10px;\n        cursor: pointer;\n        font-size: 0.85rem;\n      }\n      .modal__button:disabled {\n        opacity: 0.6;\n        cursor: not-allowed;\n      }\n      .modal__result {\n        font-size: 0.8rem;\n        color: #334155;\n        text-align: right;\n      }\n    </style>\n    <div id="drilldownModal" class="modal">\n      <div class="modal__content" role="dialog" aria-modal="true" aria-labelledby="drilldownTitle">\n        <div class="modal__header">\n          <div id="drilldownTitle" class="modal__title">Samples</div>\n          <button id="drilldownClose" class="modal__close" type="button">×</button>\n        </div>\n        <div id="drilldownStatus" class="modal__status"></div>\n        <ul id="drilldownList" class="modal__list"></ul>\n      </div>\n    </div>\n\n    <script>\n      (function () {\n        const jobId = "__JOB_ID__";\n        const dataset = "__DATASET__";\n        const modal = document.getElementById("drilldownModal");\n        const list = document.getElementById("drilldownList");\n        const status = document.getElementById("drilldownStatus");\n        const title = document.getElementById("drilldownTitle");\n        const closeBtn = document.getElementById("drilldownClose");\n\n        function openModal() {\n          modal.classList.add("open");\n        }\n\n        function closeModal() {\n          modal.classList.remove("open");\n        }\n\n        closeBtn.addEventListener("click", closeModal);\n        modal.addEventListener("click", (event) => {\n          if (event.target === modal) {\n            closeModal();\n          }\n        });\n\n        function renderList(items) {\n          list.innerHTML = "";\n          if (!items.length) {\n            status.textContent = "No samples found for this bin/outcome.";\n            return;\n          }\n          status.textContent = `Found ${items.length} samples.`;\n          const frag = document.createDocumentFragment();\n          for (const item of items) {\n            const li = document.createElement("li");\n            li.className = "modal__item";\n\n            const details = document.createElement("div");\n            details.className = "modal__details";\n\n            const link = document.createElement("a");\n            link.textContent = item.image_path || "(unknown path)";\n            if (item.image_path) {\n              const href = `/api/datasets/${encodeURIComponent(dataset)}/image?path=${encodeURIComponent(item.image_path)}`;\n              link.href = href;\n              link.target = "_blank";\n              link.rel = "noopener";\n            }\n\n            const meta = document.createElement("div");\n            meta.className = "modal__meta";\n            const time = item.time_total_min_sec != null ? item.time_total_min_sec.toFixed(6) : "n/a";\n            meta.textContent = `time=${time}s`;\n\n            details.appendChild(link);\n            details.appendChild(meta);\n\n            const action = document.createElement("div");\n            action.className = "modal__action";\n\n            const btn = document.createElement("button");\n            btn.type = "button";\n            btn.className = "modal__button";\n            btn.textContent = "Run on this image";\n\n            const result = document.createElement("div");\n            result.className = "modal__result";\n\n            btn.addEventListener("click", () => runSingle(item.image_path, btn, result));\n\n            action.appendChild(btn);\n            action.appendChild(result);\n\n            li.appendChild(details);\n            li.appendChild(action);\n            frag.appendChild(li);\n          }\n          list.appendChild(frag);\n        }\n\n        async function runSingle(imagePath, button, resultEl) {\n          if (!imagePath) {\n            resultEl.textContent = "Missing image path.";\n            return;\n          }\n          button.disabled = true;\n          resultEl.textContent = "Running...";\n          try {\n            const resp = await fetch(`/api/jobs/${jobId}/run_single`, {\n              method: "POST",\n              headers: { "Content-Type": "application/json" },\n              body: JSON.stringify({ image_path: imagePath }),\n            });\n            const data = await resp.json();\n            if (!resp.ok) {\n              resultEl.textContent = data && data.error ? data.error : "Failed.";\n              return;\n            }\n            const time = data.time_total_min_sec != null ? data.time_total_min_sec.toFixed(6) : "n/a";\n            const acc = data.accuracy != null ? data.accuracy.toFixed(3) : "n/a";\n            resultEl.textContent = `acc=${acc}, time=${time}s, best=${data.decoded_best || ""}`;\n          } catch (err) {\n            resultEl.textContent = "Request failed.";\n          } finally {\n            button.disabled = false;\n          }\n        }\n\n        async function fetchSamples(binValue, outcome) {\n          title.textContent = `Samples: bin ${binValue} / ${outcome}`;\n          status.textContent = "Loading...";\n          list.innerHTML = "";\n          openModal();\n          try {\n            const resp = await fetch(`/api/jobs/${jobId}/samples?bin=${encodeURIComponent(binValue)}&outcome=${encodeURIComponent(outcome)}`);\n            const data = await resp.json();\n            if (!resp.ok) {\n              status.textContent = data && data.error ? data.error : "Failed to load samples.";\n              return;\n            }\n            if (!Array.isArray(data)) {\n              status.textContent = "Unexpected response format.";\n              return;\n            }\n            renderList(data);\n          } catch (err) {\n            status.textContent = "Request failed.";\n          }\n        }\n\n        function attachPlotlyHandler() {\n          const plot = document.querySelector(".plotly-graph-div");\n          if (!plot || !plot.on) {\n            return;\n          }\n          plot.on("plotly_click", (event) => {\n            if (!event || !event.points || !event.points.length) {\n              return;\n            }\n            const point = event.points[0];\n            const traceName = (point.data && point.data.name ? point.data.name : "").toLowerCase();\n            let outcome = null;\n            if (traceName.includes("failed")) {\n              outcome = "failed";\n            } else if (traceName.includes("correct")) {\n              outcome = "correct";\n            }\n            if (!outcome) {\n              return;\n            }\n            const binValue = parseInt(point.x, 10);\n            if (!Number.isFinite(binValue)) {\n              return;\n            }\n            fetchSamples(binValue, outcome);\n          });\n        }\n\n        attachPlotlyHandler();\n      })();\n    </script>'
    injection = template.replace("__JOB_ID__", job_id).replace("__DATASET__", dataset)
    return html.replace("</body>", injection + "</body>")



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
