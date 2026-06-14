from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .web_visualizer_core import (
    build_training_command,
    discover_configs,
    discover_runs,
    format_training_log_tail,
    load_prediction_sample,
    load_run_payload,
    materialize_training_config,
    parse_training_progress,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS: dict[str, dict[str, Any]] = {}
LOG_PROGRESS_PLACEHOLDER = "训练进度已移到上方动态条；普通日志产生后会显示在这里。"
TRAINING_FINISHED_TEXT = "训练已完成。"


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PKR-MoE 可视化</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #66717c;
      --line: #d9dee4;
      --accent: #24735a;
      --accent-2: #b4462f;
      --warn: #b98213;
      --blue: #2f6fb4;
      --shadow: 0 10px 28px rgba(20, 32, 38, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    header h1 {
      font-size: 18px;
      margin: 0;
      letter-spacing: 0;
    }
    header .status {
      font-size: 13px;
      color: var(--muted);
      max-width: 45vw;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      padding: 16px;
      overflow: auto;
    }
    .workspace {
      padding: 18px;
      overflow: auto;
    }
    .section {
      margin-bottom: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
    }
    .panel h2,
    .panel h3,
    aside h2,
    aside h3 {
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 10px 0 5px;
    }
    select, input, button {
      width: 100%;
      min-height: 34px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 7px 9px;
      font: inherit;
      font-size: 13px;
    }
    button {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary {
      background: #fff;
      color: var(--ink);
      border-color: var(--line);
    }
    button.danger {
      background: var(--accent-2);
      border-color: var(--accent-2);
      color: #fff;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .grid {
      display: grid;
      gap: 14px;
    }
    .grid.two {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .grid.three {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .metric {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }
    .metric .label {
      font-size: 12px;
      color: var(--muted);
      margin: 0 0 8px;
      text-transform: uppercase;
    }
    .metric .value {
      font-size: 24px;
      font-weight: 700;
      white-space: nowrap;
    }
    .subtle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .run-list {
      display: grid;
      gap: 8px;
      max-height: 300px;
      overflow: auto;
    }
    .run-item {
      width: 100%;
      text-align: left;
      color: var(--ink);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font-weight: 500;
    }
    .run-item.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .tag-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .tag {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.7fr);
      gap: 14px;
    }
    .canvas-wrap {
      width: 100%;
      min-height: 360px;
    }
    canvas {
      width: 100%;
      height: 340px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fbfcfd;
    }
    .heatmap {
      display: grid;
      gap: 6px;
    }
    .heat-row {
      display: grid;
      grid-template-columns: 70px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .heat-cells {
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(48px, 1fr);
      gap: 4px;
    }
    .heat-cell {
      min-height: 32px;
      border-radius: 6px;
      border: 1px solid rgba(0,0,0,.07);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      color: #102018;
    }
    .log-box {
      height: min(42vh, 360px);
      min-height: 240px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #101820;
      color: #e6eef2;
      padding: 10px;
      font: 12px Consolas, monospace;
      white-space: pre-wrap;
    }
    .progress-wrap {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .progress-line {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .progress-track {
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: #e7ecef;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 180ms ease;
    }
    .progress-meta {
      margin-top: 8px;
      color: var(--ink);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .prediction-controls {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1.1fr) minmax(180px, 1fr);
      gap: 12px;
      align-items: start;
    }
    .control-group {
      min-width: 0;
      display: grid;
      grid-template-rows: 20px 34px 18px;
      gap: 4px;
      align-items: center;
    }
    .control-group label {
      margin: 0;
      align-self: end;
    }
    .control-label-spacer {
      display: block;
      height: 20px;
    }
    .sample-picker {
      display: grid;
      grid-template-columns: 36px minmax(72px, 1fr) 36px;
      gap: 6px;
      align-items: stretch;
    }
    .sample-picker input {
      min-width: 0;
      text-align: center;
    }
    .sample-arrow {
      width: 36px;
      min-width: 36px;
      padding: 0;
      font-size: 20px;
      line-height: 1;
    }
    .sample-index {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .empty {
      min-height: 260px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: #fff;
      text-align: center;
      padding: 24px;
    }
    @media (max-width: 980px) {
      main, .layout, .grid.two, .grid.three, .prediction-controls {
        grid-template-columns: 1fr;
      }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      header .status {
        max-width: 40vw;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>PKR-MoE 训练与推理可视化</h1>
    <div class="status" id="status">正在加载...</div>
  </header>
  <main>
    <aside>
      <section class="section panel">
        <h2>启动现有训练入口</h2>
        <label for="configSelect">任务配置</label>
        <select id="configSelect"></select>
        <label for="predLenInput">预测步长</label>
        <input id="predLenInput" type="number" min="1" step="1" />
        <label for="deviceInput">设备</label>
        <input id="deviceInput" value="cuda:0" />
        <label for="sampleCountInput">推理回放样本数</label>
        <input id="sampleCountInput" type="number" min="1" max="512" value="32" />
        <button id="startTrainBtn" style="margin-top:12px">生成诊断配置并启动训练</button>
        <p class="subtle">只调用现有 <code>src.train</code>，不修改训练或模型逻辑。</p>
      </section>
      <section class="section panel">
        <h2>已有结果</h2>
        <button class="secondary" id="refreshBtn">刷新结果列表</button>
        <div class="run-list" id="runList" style="margin-top:10px"></div>
      </section>
    </aside>
    <div class="workspace">
      <button class="secondary" id="showJobPanelBtn" style="display:none;margin-bottom:12px">显示训练窗口</button>
      <section class="section panel" id="jobPanel" style="display:none">
        <h2>训练进程</h2>
        <div class="grid three">
          <div class="metric"><div class="label">状态</div><div class="value" id="jobState">-</div></div>
          <div class="metric"><div class="label">输出目录</div><div class="value" id="jobRunDir" style="font-size:13px;white-space:normal">-</div></div>
          <div class="metric"><div class="label">退出码</div><div class="value" id="jobReturnCode">-</div></div>
        </div>
        <div class="grid two" style="margin-top:12px">
          <button class="danger" id="stopTrainBtn">停止训练</button>
          <button class="secondary" id="clearJobBtn">隐藏训练窗口</button>
        </div>
        <div class="progress-wrap" id="jobProgress" style="display:none">
          <div class="progress-line">
            <span id="jobProgressTitle">Epoch -</span>
            <span id="jobProgressPercent">-</span>
          </div>
          <div class="progress-track"><div class="progress-fill" id="jobProgressFill"></div></div>
          <div class="progress-meta" id="jobProgressMeta"></div>
        </div>
        <label>日志尾部</label>
        <div class="log-box" id="jobLog"></div>
      </section>
      <section class="section" id="content">
        <div class="empty">请选择左侧已有结果。包含 <code>prediction_intermediates.npz</code> 的 run 可以播放推理与 MoE 参与过程。</div>
      </section>
    </div>
  </main>
  <script>
    const state = {
      configs: [],
      runs: [],
      run: null,
      prediction: null,
      selectedRun: "",
      selectedChannel: 0,
      selectedSample: 0,
      jobId: "",
      jobTimer: null,
    };

    const $ = (id) => document.getElementById(id);

    function setStatus(text) {
      $("status").textContent = text;
    }

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      if (!res.ok) {
        let detail = await res.text();
        try { detail = JSON.parse(detail).error || detail; } catch (_) {}
        throw new Error(detail || `${res.status} ${res.statusText}`);
      }
      return res.json();
    }

    function fmt(value, digits = 4) {
      if (value === null || value === undefined) return "-";
      if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : "-";
      if (typeof value === "string") return value || "-";
      if (Array.isArray(value)) return value.length ? value.join(", ") : "-";
      return String(value);
    }

    function optionLabel(cfg) {
      const horizon = cfg.pred_len ? `H${cfg.pred_len}` : "H?";
      return `${cfg.dataset} ${horizon} · ${cfg.name}`;
    }

    async function loadInitial() {
      const [configsData, runsData] = await Promise.all([api("/api/configs"), api("/api/runs")]);
      state.configs = configsData.configs || [];
      state.runs = runsData.runs || [];
      renderConfigs();
      renderRuns();
      await attachActiveJob();
      setStatus(`已加载 ${state.configs.length} 个配置，${state.runs.length} 个结果`);
    }

    async function attachActiveJob() {
      const payload = await api("/api/train/active");
      const jobs = payload.jobs || [];
      const job = jobs.find(item => ["running", "stopping"].includes(item.state));
      if (!job) return;
      state.jobId = job.job_id;
      showTrainingPanel();
      $("stopTrainBtn").disabled = job.state !== "running";
      renderJobStatus(job);
      if (state.jobTimer) clearInterval(state.jobTimer);
      state.jobTimer = setInterval(pollJob, 2500);
    }

    function renderConfigs() {
      const select = $("configSelect");
      select.innerHTML = "";
      state.configs.forEach((cfg, index) => {
        const opt = document.createElement("option");
        opt.value = cfg.id;
        opt.textContent = optionLabel(cfg);
        select.appendChild(opt);
        if (index === 0) {
          $("predLenInput").value = cfg.pred_len || 96;
          $("deviceInput").value = cfg.device || "cuda:0";
        }
      });
      select.onchange = () => {
        const cfg = state.configs.find(item => item.id === select.value);
        if (!cfg) return;
        $("predLenInput").value = cfg.pred_len || 96;
        $("deviceInput").value = cfg.device || "cuda:0";
      };
    }

    function renderRuns() {
      const root = $("runList");
      root.innerHTML = "";
      if (!state.runs.length) {
        root.innerHTML = "<div class='subtle'>没有找到 run_summary.json。</div>";
        return;
      }
      state.runs.forEach(run => {
        const btn = document.createElement("button");
        btn.className = "run-item" + (state.selectedRun === run.id ? " active" : "");
        btn.innerHTML = `
          <div>${run.label}</div>
          <div class="subtle">${run.dataset || "-"} ${run.pred_len ? "H" + run.pred_len : ""} · MSE ${fmt(run.avg_mse)} · MAE ${fmt(run.avg_mae)}</div>
          <div class="tag-row"><span class="tag">${run.has_prediction_intermediates ? "可回放推理" : "无推理回放"}</span></div>
        `;
        btn.onclick = () => loadRun(run.id);
        root.appendChild(btn);
      });
    }

    async function refreshRuns() {
      const data = await api("/api/runs");
      state.runs = data.runs || [];
      renderRuns();
      setStatus("结果列表已刷新");
    }

    async function loadRun(runId) {
      setStatus("正在读取结果...");
      state.selectedRun = runId;
      renderRuns();
      state.run = await api(`/api/run?run_dir=${encodeURIComponent(runId)}`);
      state.prediction = null;
      state.selectedSample = 0;
      state.selectedChannel = 0;
      renderRun();
      if (state.run.has_prediction_intermediates) {
        await loadPrediction(0);
      }
      setStatus(`已加载 ${runId}`);
    }

    async function loadPrediction(sampleIndex) {
      state.selectedSample = sampleIndex;
      state.prediction = await api(`/api/prediction?run_dir=${encodeURIComponent(state.selectedRun)}&sample=${sampleIndex}`);
      renderRun();
    }

    async function changeSample(delta) {
      const pred = state.prediction;
      if (!pred) return;
      const maxIndex = Math.max(0, (pred.sample_count || 1) - 1);
      const nextIndex = Math.max(0, Math.min(maxIndex, state.selectedSample + delta));
      if (nextIndex === state.selectedSample) return;
      await loadPrediction(nextIndex);
    }

    function metricCards(summary) {
      const selected = summary.selected || {};
      const test = summary.test || {};
      const val = summary.val || {};
      const timing = summary.timing || {};
      const values = [
        ["Selected MSE", selected.avg_mse ?? test.avg_mse ?? val.avg_mse],
        ["Selected MAE", selected.avg_mae ?? test.avg_mae ?? val.avg_mae],
        ["Best Epoch", Array.isArray(summary.best_epoch) ? summary.best_epoch.join(", ") : "-"],
        ["Total Sec", timing.total_sec],
        ["MoE Variant", (summary.selected || {}).moe_residual_variant || "-"],
        ["Router", (summary.moe_router || {}).mode || "-"],
      ];
      return `<div class="grid three">${values.map(([label, value]) => `
        <div class="metric"><div class="label">${label}</div><div class="value">${fmt(value)}</div></div>
      `).join("")}</div>`;
    }

    function residualSummary(summary) {
      const residual = summary.moe_residual || {};
      const effective = residual.effective_route_by_penalty || {};
      const alpha = residual.alpha_by_penalty || {};
      const rows = Object.keys({...effective, ...alpha}).map(name => {
        const eff = Number(effective[name] ?? 0);
        const pct = Math.max(0, Math.min(100, eff * 100));
        return `<tr>
          <td>${name}</td>
          <td><div style="height:10px;background:#e7ecef;border-radius:999px;overflow:hidden"><div style="width:${pct}%;height:100%;background:var(--accent)"></div></div></td>
          <td>${fmt(effective[name])}</td>
          <td>${fmt(alpha[name])}</td>
        </tr>`;
      }).join("");
      return `
        <div class="panel">
          <h3>MoE Residual 参与汇总</h3>
          <div class="tag-row">
            <span class="tag">enabled=${Boolean(residual.enabled)}</span>
            <span class="tag">branch_rms=${fmt(residual.branch_rms)}</span>
            <span class="tag">residual/base=${fmt(residual.residual_base_rms_ratio)}</span>
          </div>
          <table style="margin-top:10px"><thead><tr><th>Penalty</th><th>Effective Route</th><th>值</th><th>Alpha</th></tr></thead><tbody>${rows || "<tr><td colspan='4'>无 MoE residual 汇总</td></tr>"}</tbody></table>
        </div>
      `;
    }

    function clusterPenaltyTable(rows) {
      const body = (rows || []).slice(0, 100).map(row => `
        <tr>
          <td>${row.cluster ?? row.cluster_id ?? "-"}</td>
          <td>${row.penalty ?? row.penalty_name ?? "-"}</td>
          <td>${fmt(row.avg_prob ?? row.prob ?? row.avg_gate_prob)}</td>
          <td>${fmt(row.active_rate ?? row.activation ?? row.avg_skip_active)}</td>
          <td>${fmt(row.avg_lambda)}</td>
        </tr>
      `).join("");
      return `
        <div class="panel">
          <h3>Cluster-Penalty 概率表</h3>
          <table><thead><tr><th>Cluster</th><th>Penalty</th><th>Avg Prob</th><th>Active/Skip</th><th>Lambda</th></tr></thead><tbody>${body || "<tr><td colspan='5'>未找到 cluster_penalty_probs.csv</td></tr>"}</tbody></table>
        </div>
      `;
    }

    function predictionControls(pred) {
      const channels = pred.channel_names || [];
      const sampleMax = Math.max(0, (pred.sample_count || 1) - 1);
      const prevDisabled = state.selectedSample <= 0 ? "disabled" : "";
      const nextDisabled = state.selectedSample >= sampleMax ? "disabled" : "";
      const channelOpts = channels.map((name, i) => `<option value="${i}" ${i === state.selectedChannel ? "selected" : ""}>${name} · C${pred.channel_cluster_ids?.[i] ?? "-"}</option>`).join("");
      return `
        <div class="prediction-controls">
          <div class="control-group">
            <label>样本</label>
            <div class="sample-picker">
              <button class="secondary sample-arrow" id="prevSampleBtn" title="上一个样本" aria-label="上一个样本" ${prevDisabled}>‹</button>
              <input id="sampleInput" type="number" min="0" max="${sampleMax}" value="${state.selectedSample}">
              <button class="secondary sample-arrow" id="nextSampleBtn" title="下一个样本" aria-label="下一个样本" ${nextDisabled}>›</button>
            </div>
            <div class="sample-index">窗口 idx=${pred.idx ?? "-"}</div>
          </div>
          <div class="control-group">
            <label>通道</label>
            <select id="channelSelect">${channelOpts}</select>
            <div></div>
          </div>
          <div class="control-group">
            <span class="control-label-spacer"></span>
            <button class="secondary" id="reloadSampleBtn">加载样本</button>
            <div></div>
          </div>
        </div>
      `;
    }

    function heatColor(value) {
      const v = Math.max(0, Math.min(1, Number(value) || 0));
      const r = Math.round(246 - 190 * v);
      const g = Math.round(248 - 84 * v);
      const b = Math.round(246 - 108 * v);
      return `rgb(${r},${g},${b})`;
    }

    function gateHeatmap(pred) {
      const penalties = pred.penalty_names || [];
      const header = `<div class="heat-row"><div></div><div class="heat-cells">${penalties.map(p => `<div class="subtle" style="text-align:center">${p}</div>`).join("")}</div></div>`;
      const rows = (pred.clusters || []).map(cluster => `
        <div class="heat-row">
          <div>C${cluster.cluster}<br><span class="subtle">top ${cluster.top_penalty || "-"}</span></div>
          <div class="heat-cells">
            ${(cluster.probabilities || []).map(p => `<div class="heat-cell" style="background:${heatColor(p.prob)}">${fmt(p.prob, 2)}</div>`).join("")}
          </div>
        </div>
      `).join("");
      return `
        <div class="panel">
          <h3>当前样本 MoE Gate</h3>
          <div class="heatmap">${header}${rows || "<div class='subtle'>无 gate_probs。</div>"}</div>
          <table style="margin-top:12px"><thead><tr><th>Cluster</th><th>Penalty</th><th>参与</th><th>Gate强度</th><th>有效强度</th><th>Skip</th></tr></thead><tbody>
            ${(pred.clusters || []).flatMap(c => (c.penalty_participation || []).map(p => `
              <tr>
                <td>${c.cluster}</td>
                <td>${p.penalty}</td>
                <td>${p.selected ? "是" : "否"}</td>
                <td>${fmt(p.gate_prob)}</td>
                <td>${fmt(p.effective_strength)}</td>
                <td>${fmt(c.skip_prob)}</td>
              </tr>
            `)).join("") || "<tr><td colspan='6'>无 penalty 参与明细</td></tr>"}
          </tbody></table>
          <p class="subtle">参与来自 gate_mask；Gate强度来自 gate_probs；有效强度 = 参与 * Gate强度 * (1 - Skip)。</p>
        </div>
      `;
    }

    function renderRun() {
      if (!state.run) return;
      const summary = state.run.summary || {};
      const pred = state.prediction;
      const replay = pred ? `
        <div class="panel">
          <h3>推理回放</h3>
          ${predictionControls(pred)}
          <div class="canvas-wrap" style="margin-top:12px"><canvas id="curveCanvas"></canvas></div>
          <div class="tag-row" id="legendTags"></div>
        </div>
      ` : `<div class="panel"><h3>推理回放</h3><div class="empty">这个 run 没有 prediction_intermediates.npz，无法播放推理过程。</div></div>`;
      $("content").innerHTML = `
        <div class="section panel">
          <h2>${state.selectedRun}</h2>
          <div class="tag-row">
            ${(summary.penalty_names || []).map(name => `<span class="tag">${name}</span>`).join("")}
          </div>
          <div style="margin-top:12px">${metricCards(summary)}</div>
        </div>
        <div class="section layout">
          <div class="grid">
            ${replay}
            ${clusterPenaltyTable(state.run.cluster_penalty_rows)}
          </div>
          <div class="grid">
            ${residualSummary(summary)}
            ${pred ? gateHeatmap(pred) : ""}
          </div>
        </div>
      `;
      if (pred) {
        $("channelSelect").onchange = (event) => {
          state.selectedChannel = Number(event.target.value);
          drawCurves();
        };
        $("reloadSampleBtn").onclick = async () => {
          const maxIndex = Math.max(0, (pred.sample_count || 1) - 1);
          const index = Math.max(0, Math.min(maxIndex, Number($("sampleInput").value || 0)));
          await loadPrediction(index);
        };
        $("prevSampleBtn").onclick = async () => changeSample(-1);
        $("nextSampleBtn").onclick = async () => changeSample(1);
        drawCurves();
      }
    }

    function drawCurves() {
      const pred = state.prediction;
      if (!pred) return;
      const canvas = $("curveCanvas");
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(600, Math.floor(rect.width * scale));
      canvas.height = Math.floor(340 * scale);
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      const W = canvas.width / scale;
      const H = canvas.height / scale;
      ctx.clearRect(0, 0, W, H);
      const pad = {left: 46, right: 18, top: 18, bottom: 34};
      const c = state.selectedChannel;
      const seriesMap = {
        "true": {data: pred.series.y_true?.[c] || [], color: "#172026"},
        "base": {data: pred.series.y_base?.[c] || [], color: "#2f6fb4"},
        "raw residual": {data: pred.series.y_residual_raw?.[c] || [], color: "#b98213"},
        "final": {data: pred.series.y_final?.[c] || [], color: "#24735a"},
      };
      const all = Object.values(seriesMap).flatMap(item => item.data.map(Number)).filter(Number.isFinite);
      if (!all.length) return;
      const min = Math.min(...all);
      const max = Math.max(...all);
      const span = Math.max(1e-9, max - min);
      const xMax = Math.max(1, Math.max(...Object.values(seriesMap).map(item => item.data.length)) - 1);
      const x = (i) => pad.left + (i / xMax) * (W - pad.left - pad.right);
      const y = (v) => pad.top + (1 - ((v - min) / span)) * (H - pad.top - pad.bottom);
      ctx.strokeStyle = "#d9dee4";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 4; i++) {
        const yy = pad.top + (i / 4) * (H - pad.top - pad.bottom);
        ctx.moveTo(pad.left, yy);
        ctx.lineTo(W - pad.right, yy);
      }
      ctx.stroke();
      ctx.fillStyle = "#66717c";
      ctx.font = "12px Segoe UI";
      ctx.fillText(max.toFixed(2), 6, pad.top + 4);
      ctx.fillText(min.toFixed(2), 6, H - pad.bottom);
      Object.entries(seriesMap).forEach(([name, item]) => {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = name === "true" ? 2.2 : 1.7;
        ctx.beginPath();
        item.data.forEach((v, i) => {
          const xx = x(i);
          const yy = y(Number(v));
          if (i === 0) ctx.moveTo(xx, yy);
          else ctx.lineTo(xx, yy);
        });
        ctx.stroke();
      });
      const legend = $("legendTags");
      if (legend) {
        legend.innerHTML = Object.entries(seriesMap).map(([name, item]) => `<span class="tag"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${item.color};margin-right:5px"></span>${name}</span>`).join("");
      }
    }

    async function startTraining() {
      const body = {
        config_id: $("configSelect").value,
        pred_len: Number($("predLenInput").value || 0),
        device: $("deviceInput").value || "cuda:0",
        sample_count: Number($("sampleCountInput").value || 32),
      };
      $("startTrainBtn").disabled = true;
      try {
        const payload = await api("/api/train/start", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body),
        });
        state.jobId = payload.job_id;
        showTrainingPanel();
        $("stopTrainBtn").disabled = false;
        pollJob();
        if (state.jobTimer) clearInterval(state.jobTimer);
        state.jobTimer = setInterval(pollJob, 2500);
        setStatus(`训练已启动：${payload.run_dir}`);
      } finally {
        $("startTrainBtn").disabled = false;
      }
    }

    async function stopTraining() {
      if (!state.jobId) return;
      $("stopTrainBtn").disabled = true;
      await api("/api/train/stop", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({job_id: state.jobId}),
      });
      await pollJob();
    }

    async function pollJob() {
      if (!state.jobId) return;
      const job = await api(`/api/train/status?job_id=${encodeURIComponent(state.jobId)}`);
      renderJobStatus(job);
      if (!["running", "stopping"].includes(job.state) && state.jobTimer) {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        $("stopTrainBtn").disabled = true;
        await refreshRuns();
      }
    }

    function renderJobStatus(job) {
      $("jobState").textContent = job.state || "-";
      $("jobRunDir").textContent = job.run_dir || "-";
      $("jobReturnCode").textContent = job.returncode === null || job.returncode === undefined ? "-" : job.returncode;
      renderJobProgress(job.progress || {});
      $("jobLog").textContent = job.log_tail || "";
    }

    function showTrainingPanel() {
      $("jobPanel").style.display = "block";
      $("showJobPanelBtn").style.display = "none";
    }

    function hideTrainingPanel() {
      $("jobPanel").style.display = "none";
      if (state.jobId) {
        $("showJobPanelBtn").style.display = "block";
      }
    }

    function renderJobProgress(progress) {
      const root = $("jobProgress");
      if (!progress || Object.keys(progress).length === 0) {
        root.style.display = "none";
        return;
      }
      root.style.display = "block";
      const epoch = progress.epoch_total
        ? `Epoch ${progress.epoch_current}/${progress.epoch_total}`
        : `Epoch ${progress.epoch_current}`;
      const batch = progress.batch_total
        ? `Batch ${progress.batch_current}/${progress.batch_total}`
        : "";
      const percent = Number(progress.display_percent ?? progress.global_percent ?? progress.epoch_percent ?? 0);
      const safePercent = Math.max(0, Math.min(100, Number.isFinite(percent) ? percent : 0));
      $("jobProgressTitle").textContent = [epoch, batch, progress.phase].filter(Boolean).join(" · ");
      $("jobProgressPercent").textContent = `${safePercent.toFixed(1)}%`;
      $("jobProgressFill").style.width = `${safePercent}%`;
      const loss = progress.loss === null || progress.loss === undefined ? "" : `loss=${fmt(progress.loss, 6)}`;
      const valLoss = progress.val_loss === null || progress.val_loss === undefined ? "" : `val_loss=${fmt(progress.val_loss, 6)}`;
      const epochPercent = progress.epoch_percent === null || progress.epoch_percent === undefined ? "" : `epoch=${fmt(progress.epoch_percent, 1)}%`;
      $("jobProgressMeta").textContent = [epochPercent, loss, valLoss, progress.raw].filter(Boolean).join(" · ");
    }

    $("refreshBtn").onclick = refreshRuns;
    $("startTrainBtn").onclick = startTraining;
    $("stopTrainBtn").onclick = stopTraining;
    $("clearJobBtn").onclick = hideTrainingPanel;
    $("showJobPanelBtn").onclick = showTrainingPanel;
    window.addEventListener("resize", () => { if (state.prediction) drawCurves(); });
    loadInitial().catch(err => {
      console.error(err);
      setStatus(`加载失败：${err.message}`);
      $("content").innerHTML = `<div class="empty">加载失败：${err.message}</div>`;
    });
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, text: str, content_type: str = "text/html; charset=utf-8") -> None:
    body = text.encode("utf-8")
    handler.send_response(int(HTTPStatus.OK))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _tail(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _append_log_line(path: Path, text: str) -> None:
    with path.open("ab") as f:
        f.write(("\n" + text.rstrip() + "\n").encode("utf-8", errors="replace"))


class _PidProcess:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)

    def poll(self) -> int | None:
        return 0 if not _pid_running(self.pid) else None


def _pid_running(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return str(int(pid)) in result.stdout
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _terminate_process_tree(proc: subprocess.Popen[Any], extra_pids: list[int] | None = None) -> None:
    pids = [int(proc.pid)] + [int(pid) for pid in (extra_pids or [])]
    pids = list(dict.fromkeys(pid for pid in pids if pid > 0))
    if not pids:
        return
    if os.name == "nt":
        for pid in pids:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _find_running_train_pids(config_path: Path) -> list[int]:
    if os.name != "nt":
        return []
    target = config_path.as_posix().replace("'", "''")
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*src.train*' -and "
        f"$_.CommandLine -like '*{target}*' -and "
        "$_.CommandLine -notlike '*Get-CimInstance*' -and "
        "$_.Name -notlike '*powershell*' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return list(dict.fromkeys(pids))


def _adopt_job(job_id: str) -> dict[str, Any] | None:
    run_dir = REPO_ROOT / "outputs" / "web_visualizer" / "runs" / job_id
    config_path = run_dir / "config.yaml"
    log_path = run_dir / "train.log"
    if not config_path.exists():
        return None
    pids = _find_running_train_pids(config_path)
    if not pids:
        return None
    job = {
        "job_id": job_id,
        "process": _PidProcess(pids[0]),
        "pids": pids,
        "cmd": ["adopted", "src.train", "--config", config_path.as_posix()],
        "config_path": config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "log_path": log_path.as_posix(),
        "started_at": time.time(),
        "adopted": True,
    }
    JOBS[job_id] = job
    return job


def _active_training_jobs() -> list[dict[str, Any]]:
    runs_root = REPO_ROOT / "outputs" / "web_visualizer" / "runs"
    if not runs_root.exists():
        return []
    jobs = []
    for config_path in sorted(runs_root.glob("*/config.yaml"), key=lambda path: path.stat().st_mtime, reverse=True):
        pids = _find_running_train_pids(config_path)
        if not pids:
            continue
        job_id = config_path.parent.name
        if job_id not in JOBS:
            _adopt_job(job_id)
        try:
            status = _job_status(job_id)
        except Exception:
            continue
        if status.get("state") in {"running", "stopping"}:
            jobs.append(status)
    return jobs


def _start_training(payload: dict[str, Any]) -> dict[str, Any]:
    config_id = str(payload.get("config_id") or "")
    if not config_id:
        raise ValueError("config_id is required")
    pred_len = payload.get("pred_len")
    pred_len_int = int(pred_len) if pred_len not in (None, "") else None
    config_name = Path(config_id).stem
    run_id = f"{config_name}-{pred_len_int or 'h'}-{int(time.time())}"
    materialized = materialize_training_config(
        REPO_ROOT,
        config_id,
        run_id=run_id,
        pred_len=pred_len_int,
        device=str(payload.get("device") or "cuda:0"),
        sample_count=int(payload.get("sample_count") or 32),
    )
    run_dir = Path(materialized["run_dir"])
    log_path = run_dir / "train.log"
    log_file = log_path.open("wb")
    cmd = build_training_command(materialized["config_path"])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MOELOSS_PROGRESS_FORCE"] = "1"
    env.setdefault("MOELOSS_PROGRESS_LEAVE", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    log_file.close()
    job_id = materialized["run_id"]
    JOBS[job_id] = {
        "job_id": job_id,
        "process": proc,
        "cmd": cmd,
        "config_path": materialized["config_path"],
        "run_dir": materialized["run_dir"],
        "log_path": log_path.as_posix(),
        "started_at": time.time(),
    }
    return {
        "job_id": job_id,
        "state": "running",
        "cmd": cmd,
        "config_path": materialized["config_path"],
        "run_dir": materialized["run_dir"],
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _progress_for_state(raw_log_tail: str, state: str) -> dict[str, Any]:
    progress = dict(parse_training_progress(raw_log_tail))
    if state == "finished":
        progress.update(
            {
                "phase": "finished",
                "epoch_percent": 100.0,
                "global_percent": 100.0,
                "display_percent": 100.0,
                "raw": TRAINING_FINISHED_TEXT,
            }
        )
        return progress

    global_percent = _number(progress.get("global_percent"))
    epoch_percent = _number(progress.get("epoch_percent"))
    if global_percent is not None:
        progress["display_percent"] = global_percent
    elif epoch_percent is not None:
        progress["display_percent"] = epoch_percent
    return progress


def _log_tail_for_state(raw_log_tail: str, state: str) -> str:
    text = format_training_log_tail(raw_log_tail)
    if text:
        return text
    if state in {"running", "stopping"}:
        return LOG_PROGRESS_PLACEHOLDER
    if state == "finished":
        return TRAINING_FINISHED_TEXT
    return "暂无普通日志。"


def _job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id) or _adopt_job(job_id)
    if not job:
        raise KeyError(f"unknown job_id: {job_id}")
    proc = job["process"]
    returncode = None if any(_pid_running(pid) for pid in job.get("pids", [])) else proc.poll()
    if returncode is None:
        state = "stopping" if job.get("stop_requested") else "running"
    elif job.get("stop_requested"):
        state = "stopped"
    else:
        state = "finished" if returncode == 0 else "failed"
    raw_log_tail = _tail(Path(job["log_path"]), limit=12000)
    progress = _progress_for_state(raw_log_tail, state)
    return {
        "job_id": job_id,
        "state": state,
        "returncode": returncode,
        "cmd": job["cmd"],
        "config_path": job["config_path"],
        "run_dir": job["run_dir"],
        "started_at": job["started_at"],
        "log_tail": _log_tail_for_state(raw_log_tail, state),
        "progress": progress,
    }


def _stop_training(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id) or _adopt_job(job_id)
    if not job:
        raise KeyError(f"unknown job_id: {job_id}")
    proc = job["process"]
    if proc.poll() is not None:
        return _job_status(job_id)
    job["stop_requested"] = True
    job["stop_requested_at"] = time.time()
    _append_log_line(Path(job["log_path"]), "[Web] Stop requested.")
    _terminate_process_tree(proc, extra_pids=job.get("pids", []))
    return _job_status(job_id)


class VisualizerHandler(BaseHTTPRequestHandler):
    server_version = "MoEVisualizer/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/":
                _text_response(self, INDEX_HTML)
            elif parsed.path == "/api/configs":
                _json_response(self, {"configs": discover_configs(REPO_ROOT)})
            elif parsed.path == "/api/runs":
                _json_response(self, {"runs": discover_runs(REPO_ROOT)})
            elif parsed.path == "/api/run":
                run_dir = (qs.get("run_dir") or [""])[0]
                _json_response(self, load_run_payload(REPO_ROOT, run_dir))
            elif parsed.path == "/api/prediction":
                run_dir = (qs.get("run_dir") or [""])[0]
                sample = int((qs.get("sample") or ["0"])[0])
                _json_response(self, load_prediction_sample(REPO_ROOT, run_dir, sample))
            elif parsed.path == "/api/train/status":
                job_id = (qs.get("job_id") or [""])[0]
                _json_response(self, _job_status(job_id))
            elif parsed.path == "/api/train/active":
                _json_response(self, {"jobs": _active_training_jobs()})
            else:
                _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/train/start":
                payload = _read_request_json(self)
                _json_response(self, _start_training(payload))
            elif parsed.path == "/api/train/stop":
                payload = _read_request_json(self)
                _json_response(self, _stop_training(str(payload.get("job_id") or "")))
            else:
                _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local PKR-MoE visualization dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, int(args.port)), VisualizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"PKR-MoE visualizer: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
