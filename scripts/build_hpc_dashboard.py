#!/usr/bin/env python3
"""Build a lightweight HTML dashboard from stEVE training logs.

Usage:
  python3 scripts/build_hpc_dashboard.py \
    --input logs/hpc/raw \
    --out logs/hpc/dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


HEARTBEAT_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?heartbeat\[(?P<stage>\w+)\]: "
    r"heatup=(?P<heatup>\d+) \(\+(?P<dheatup>\d+)/\s*(?P<dt>\d+)s\) "
    r"explore=(?P<explore>\d+) \(\+(?P<dexplore>\d+)/\s*(?P<dt2>\d+)s\) "
    r"update=(?P<update>\d+) \(\+(?P<dupdate>\d+)/\s*(?P<dt3>\d+)s\)"
)

TRAIN_META_RE = re.compile(
    r"\[train\]\s+tool=(?P<tool>\S+)\s+trainer_device=(?P<device>\S+)\s+workers=(?P<workers>\d+)"
)


def _parse_heartbeats(log_path: Path) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = HEARTBEAT_RE.match(line)
            if not m:
                continue
            ts = m.group("ts")
            stage = m.group("stage")
            dheatup = int(m.group("dheatup"))
            dexplore = int(m.group("dexplore"))
            dupdate = int(m.group("dupdate"))
            dt = int(m.group("dt"))
            rate = 0.0
            if dt > 0:
                rate = (dheatup / dt) if stage == "heatup" else (dexplore / dt)
            update_rate = (dupdate / dt) if dt > 0 else 0.0
            points.append(
                {
                    "ts": ts,
                    "stage": stage,
                    "heatup": int(m.group("heatup")),
                    "explore": int(m.group("explore")),
                    "update": int(m.group("update")),
                    "dheatup": dheatup,
                    "dexplore": dexplore,
                    "dupdate": dupdate,
                    "dt": dt,
                    "rate": round(rate, 6),
                    "update_rate": round(update_rate, 6),
                }
            )
    return points


def _parse_train_meta(log_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TRAIN_META_RE.search(line)
            if m:
                meta["tool"] = m.group("tool")
                meta["device"] = m.group("device")
                meta["workers"] = int(m.group("workers"))
                break
    return meta


def _load_run_json(run_dir: Path) -> Dict[str, Any]:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return {}
    try:
        return json.loads(run_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_warning": f"Invalid JSON in {run_json}"}


def _load_job_sacct(run_dir: Path) -> Dict[str, Any]:
    sacct_path = run_dir / "job.sacct.txt"
    if not sacct_path.exists():
        return {}
    lines = [l.strip() for l in sacct_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < 2 or "|" not in lines[0]:
        return {}
    header = lines[0].split("|")
    values = lines[1].split("|")
    return dict(zip(header, values))


def _load_sbatch_text(run_dir: Path) -> Optional[str]:
    sbatch_path = run_dir / "job.sbatch"
    if not sbatch_path.exists():
        return None
    text = sbatch_path.read_text(encoding="utf-8", errors="ignore")
    # Keep dashboard snappy; cap size.
    if len(text) > 5000:
        return text[:5000] + "\n... (truncated)"
    return text


def _find_csv_for_run(run_dir: Path) -> Optional[Path]:
    # CSVs live next to run dir, named "<run_dir>.csv"
    candidate = run_dir.parent / f"{run_dir.name}.csv"
    if candidate.exists():
        return candidate
    return None


def _collect_worker_logs(run_dir: Path, out_root: Path, limit: int = 80) -> List[Dict[str, str]]:
    logs_dir = run_dir / "logs_subprocesses"
    if not logs_dir.exists():
        return []
    files = sorted(p for p in logs_dir.rglob("*") if p.is_file())
    out: List[Dict[str, str]] = []
    for p in files[:limit]:
        out.append(
            {
                "name": p.name,
                "href": Path(os.path.relpath(p, out_root)).as_posix(),
                "path": str(p),
            }
        )
    return out


def _to_float(val: str) -> Optional[float]:
    try:
        v = float(val)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _parse_csv_metrics(csv_path: Path) -> Dict[str, Any]:
    if not csv_path or not csv_path.exists():
        return {"evals": [], "params": {}}

    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return {"evals": [], "params": {}}

    header = [h.strip() for h in lines[0].split(";")]
    evals: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {}

    for raw in lines[1:]:
        cols = raw.split(";")
        # pad columns
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        row = {header[i]: cols[i].strip() for i in range(len(header))}

        steps_explore = row.get("steps explore", "")
        steps_val = _to_float(steps_explore)
        if steps_val is not None:
            evals.append(
                {
                    "steps": steps_val,
                    "quality": _to_float(row.get("quality", "")),
                    "reward": _to_float(row.get("reward", "")),
                    "best_quality": _to_float(row.get("best quality", "")),
                    "best_explore": _to_float(row.get("best explore steps", "")),
                }
            )
        # capture hyperparams row (often the row with lr/hidden)
        if row.get("lr"):
            params = {
                "lr": row.get("lr"),
                "hidden_layers": row.get("hidden_layers"),
                "embedder_nodes": row.get("embedder_nodes"),
                "embedder_layers": row.get("embedder_layers"),
                "HEATUP_STEPS": row.get("HEATUP_STEPS"),
                "TRAINING_STEPS": row.get("TRAINING_STEPS"),
                "EXPLORE_STEPS_BTW_EVAL": row.get("EXPLORE_STEPS_BTW_EVAL"),
                "EVAL_EPISODES": row.get("EVAL_EPISODES"),
                "CONSECUTIVE_EXPLORE_EPISODES": row.get("CONSECUTIVE_EXPLORE_EPISODES"),
                "UPDATE_PER_EXPLORE_STEP": row.get("UPDATE_PER_EXPLORE_STEP"),
                "BATCH_SIZE": row.get("BATCH_SIZE"),
                "REPLAY_BUFFER_SIZE": row.get("REPLAY_BUFFER_SIZE"),
                "REPLAY_DEVICE": row.get("REPLAY_DEVICE"),
            }

    return {"evals": evals, "params": params}


def _summarize(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not points:
        return {
            "start_ts": None,
            "end_ts": None,
            "duration_s": None,
            "last_heatup": 0,
            "last_explore": 0,
            "last_update": 0,
            "avg_heatup_rate": 0.0,
            "avg_train_rate": 0.0,
            "avg_update_rate": 0.0,
        }

    start_ts = points[0]["ts"]
    end_ts = points[-1]["ts"]
    try:
        start = datetime.strptime(start_ts, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_ts, "%Y-%m-%d %H:%M:%S")
        duration_s = int((end - start).total_seconds())
    except ValueError:
        duration_s = None

    last = points[-1]
    heat_rates = []
    train_rates = []
    update_rates = []
    for p in points:
        if p["dt"] <= 0:
            continue
        if p["stage"] == "heatup" and p["dheatup"] > 0:
            heat_rates.append(p["dheatup"] / p["dt"])
        if p["stage"] == "train" and p["dexplore"] > 0:
            train_rates.append(p["dexplore"] / p["dt"])
        if p["stage"] == "train" and p["dupdate"] > 0:
            update_rates.append(p["dupdate"] / p["dt"])
    avg_heat = sum(heat_rates) / len(heat_rates) if heat_rates else 0.0
    avg_train = sum(train_rates) / len(train_rates) if train_rates else 0.0
    avg_update = sum(update_rates) / len(update_rates) if update_rates else 0.0

    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": duration_s,
        "last_heatup": last["heatup"],
        "last_explore": last["explore"],
        "last_update": last["update"],
        "avg_heatup_rate": round(avg_heat, 3),
        "avg_train_rate": round(avg_train, 3),
        "avg_update_rate": round(avg_update, 3),
    }


def _guess_group(log_path: Path) -> str:
    parts = log_path.parts
    for key in ("paper_runs", "bench", "smoke", "results"):
        if key in parts:
            return key
    return "runs"


def _collect_logs(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("main.log"))


def build_dashboard(input_root: Path, out_root: Path) -> None:
    logs = _collect_logs(input_root)
    out_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for log in logs:
        run_dir = log.parent
        points = _parse_heartbeats(log)
        meta = _parse_train_meta(log)
        run_meta = _load_run_json(run_dir)
        job_meta = run_meta.get("job") if isinstance(run_meta, dict) else None
        if not job_meta:
            job_meta = _load_job_sacct(run_dir)
            if isinstance(run_meta, dict):
                run_meta["job"] = job_meta
        sbatch_text = _load_sbatch_text(run_dir)
        if isinstance(run_meta, dict) and sbatch_text:
            run_meta["sbatch_text"] = sbatch_text
        csv_path = _find_csv_for_run(run_dir)
        csv_meta = _parse_csv_metrics(csv_path) if csv_path else {"evals": [], "params": {}}
        summary = _summarize(points)
        log_href = Path(os.path.relpath(log, out_root)).as_posix()
        worker_logs = _collect_worker_logs(run_dir, out_root)
        run = {
            "id": str(run_dir),
            "name": run_dir.name,
            "group": _guess_group(log),
            "path": str(log),
            "log_href": log_href,
            "worker_logs": worker_logs,
            "meta": meta,
            "run_meta": run_meta,
            "job": job_meta or {},
            "csv": csv_meta,
            "summary": summary,
            "points": points,
        }
        runs.append(run)

    data = {"runs": runs}
    data_path = out_root / "data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    html_path = out_root / "index.html"
    html_path.write_text(_render_html(data), encoding="utf-8")

    print(f"[ok] wrote {data_path}")
    print(f"[ok] wrote {html_path}")


def _render_html(data: Dict[str, Any]) -> str:
    # Embed JSON directly to avoid separate fetch.
    payload = json.dumps(data)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HPC Training Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-1: #f6f8fb;
      --bg-2: #fff7ed;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #5b6472;
      --accent: #0f766e;
      --accent-2: #f59e0b;
      --border: #e2e8f0;
      --shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(800px 400px at 10% 0%, rgba(15, 118, 110, 0.08), transparent 60%),
        radial-gradient(900px 500px at 90% 10%, rgba(245, 158, 11, 0.12), transparent 60%),
        linear-gradient(135deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
    }
    .page {
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 24px 64px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    .title {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .subtitle { color: var(--muted); }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: var(--shadow);
      animation: fadeUp 0.4s ease both;
    }
    .grid { display: grid; gap: 16px; }
    .grid-2 { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .grid-3 { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .controls {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      align-items: center;
    }
    label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    input, select, button {
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      font-family: inherit;
      font-size: 13px;
    }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary {
      background: #e2e8f0;
      color: #0f172a;
    }
    .muted { color: var(--muted); }
    .tag {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      font-size: 12px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.08);
      color: var(--accent);
      margin-left: 6px;
    }
    .run-list { max-height: 220px; overflow: auto; border: 1px dashed var(--border); border-radius: 10px; padding: 8px; }
    .run-item { display: flex; align-items: center; gap: 8px; padding: 6px 4px; }
    .run-item input { width: auto; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--border); padding: 8px 6px; text-align: left; }
    th { background: #f8fafc; position: sticky; top: 0; z-index: 1; }
    canvas { max-width: 100%; }
    pre { background: #0b1220; color: #e2e8f0; padding: 12px; border-radius: 10px; overflow: auto; font-size: 12px; }
    .split { display: grid; gap: 16px; grid-template-columns: 1.2fr 1fr; }
    .metric { font-weight: 600; color: var(--accent); }
    .legend { display: flex; gap: 12px; flex-wrap: wrap; }
    .legend span { font-size: 12px; color: var(--muted); }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 900px) {
      .split { grid-template-columns: 1fr; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="page">
    <header>
      <div>
        <div class="title">HPC Training Dashboard</div>
        <div class="subtitle">Stage-aware throughput, eval quality, and job metadata across runs.</div>
      </div>
      <div class="legend">
        <span>Filters control run selection</span>
        <span>Charts compare selected runs</span>
      </div>
    </header>

    <div class="card">
      <div class="controls">
        <div>
          <label for="textFilter">Search</label>
          <input id="textFilter" type="text" placeholder="name / tool / description" />
        </div>
        <div>
          <label for="groupFilter">Group</label>
          <select id="groupFilter"></select>
        </div>
        <div>
          <label for="toolFilter">Tool</label>
          <select id="toolFilter"></select>
        </div>
        <div>
          <label for="deviceFilter">Device</label>
          <select id="deviceFilter"></select>
        </div>
        <div>
          <label for="workerFilter">Workers</label>
          <select id="workerFilter"></select>
        </div>
        <div>
          <label for="stateFilter">Job State</label>
          <select id="stateFilter"></select>
        </div>
        <div>
          <label for="stageFilter">Stage</label>
          <select id="stageFilter">
            <option value="">All stages</option>
            <option value="heatup">Heatup</option>
            <option value="train">Train</option>
          </select>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="selectAll">Select All</button>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="clearSel" class="secondary">Clear</button>
        </div>
      </div>

      <div id="runMeta" class="muted" style="margin-top:10px;"></div>
      <div class="split" style="margin-top:10px;">
        <div>
          <div class="muted" style="margin-bottom:6px;">Runs</div>
          <div id="runList" class="run-list"></div>
        </div>
        <div>
          <div class="muted" style="margin-bottom:6px;">Run details</div>
          <div id="runDetails" class="muted" style="white-space:pre-wrap;"></div>
          <div style="margin-top:8px;">
            <a id="runLogLink" href="#" target="_blank">Open main.log</a>
            <span id="runLogPath" class="muted" style="margin-left:6px;"></span>
          </div>
          <details style="margin-top:8px;">
            <summary>Show worker logs</summary>
            <div id="runWorkerLogs" class="muted"></div>
          </details>
          <details style="margin-top:8px;">
            <summary>Show sbatch</summary>
            <pre id="runSbatch"></pre>
          </details>
        </div>
      </div>
    </div>

    <div class="grid grid-3" style="margin-top: 16px;">
      <div class="card">
        <h3>Explore Steps vs Time</h3>
        <canvas id="chartExplore" height="180"></canvas>
      </div>
      <div class="card">
        <h3>Stage-Aware Steps/s</h3>
        <canvas id="chartRate" height="180"></canvas>
      </div>
      <div class="card">
        <h3>Update Steps/s</h3>
        <canvas id="chartUpdate" height="180"></canvas>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top: 16px;">
      <div class="card">
        <h3>Eval Quality vs Explore Steps</h3>
        <canvas id="chartQuality" height="220"></canvas>
      </div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h3>Runs Summary</h3>
      <div style="overflow:auto;">
        <table id="summaryTable">
          <thead>
            <tr>
              <th>Run</th>
              <th>Group</th>
              <th>Tool</th>
              <th>Device</th>
              <th>Workers</th>
              <th>State</th>
              <th>Elapsed</th>
              <th>Description</th>
              <th>Start</th>
              <th>End</th>
              <th>Heatup</th>
              <th>Explore</th>
              <th>Update</th>
              <th>Avg Heatup/s</th>
              <th>Avg Train/s</th>
              <th>Avg Update/s</th>
              <th>Eval Cnt</th>
              <th>Best Quality</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const DATA = __PAYLOAD__;
    const COLORS = ["#0f766e","#2563eb","#f59e0b","#16a34a","#ef4444","#0ea5e9","#9333ea"];

    function fmt(val) {
      return (val === null || val === undefined) ? "" : String(val);
    }
    function fmtNum(val, digits) {
      if (val === null || val === undefined || isNaN(val)) return "";
      const d = digits ?? 2;
      return Number(val).toFixed(d);
    }
    function jobMeta(run) {
      const rm = run.run_meta || {};
      return run.job || rm.job || {};
    }
    function jobState(run) {
      return jobMeta(run).State || "";
    }
    function jobElapsed(run) {
      return jobMeta(run).Elapsed || "";
    }
    function jobStart(run) {
      const job = jobMeta(run);
      return job.Start || (run.summary || {}).start_ts || "";
    }
    function jobEnd(run) {
      const job = jobMeta(run);
      return job.End || (run.summary || {}).end_ts || "";
    }
    function bestQuality(run) {
      const evals = (run.csv || {}).evals || [];
      let best = null;
      for (const e of evals) {
        const v = (e.best_quality !== null && e.best_quality !== undefined) ? e.best_quality : e.quality;
        if (v === null || v === undefined) continue;
        if (best === null || v > best) best = v;
      }
      return best;
    }

    function buildSummary(filtered) {
      const tbody = document.querySelector("#summaryTable tbody");
      tbody.innerHTML = "";
      for (const run of filtered) {
        const tr = document.createElement("tr");
        const meta = run.meta || {};
        const rmeta = run.run_meta || {};
        const s = run.summary || {};
        const evalCount = ((run.csv || {}).evals || []).length;
        const cells = [
          run.name,
          run.group,
          meta.tool || "",
          meta.device || "",
          meta.workers || "",
          jobState(run),
          jobElapsed(run),
          rmeta.description || "",
          jobStart(run),
          jobEnd(run),
          s.last_heatup || 0,
          s.last_explore || 0,
          s.last_update || 0,
          fmtNum(s.avg_heatup_rate, 2),
          fmtNum(s.avg_train_rate, 2),
          fmtNum(s.avg_update_rate, 2),
          evalCount,
          fmtNum(bestQuality(run), 3),
        ];
        for (const c of cells) {
          const td = document.createElement("td");
          td.textContent = fmt(c);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
    }

    function uniqueValues(keyFn, runs) {
      const vals = new Set();
      runs.forEach(r => {
        const v = keyFn(r);
        if (v !== undefined && v !== null && v !== "") vals.add(v);
      });
      return Array.from(vals).sort();
    }

    function setSelectOptions(sel, values, label) {
      sel.innerHTML = "";
      const all = document.createElement("option");
      all.value = "";
      all.textContent = "All " + label;
      sel.appendChild(all);
      values.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      });
    }

    function buildFilters() {
      setSelectOptions(document.getElementById("groupFilter"),
        uniqueValues(r => r.group, DATA.runs), "groups");
      setSelectOptions(document.getElementById("toolFilter"),
        uniqueValues(r => (r.meta||{}).tool, DATA.runs), "tools");
      setSelectOptions(document.getElementById("deviceFilter"),
        uniqueValues(r => (r.meta||{}).device, DATA.runs), "devices");
      setSelectOptions(document.getElementById("workerFilter"),
        uniqueValues(r => (r.meta||{}).workers, DATA.runs).map(String), "workers");
      setSelectOptions(document.getElementById("stateFilter"),
        uniqueValues(r => jobState(r), DATA.runs), "states");
    }

    function applyFilters() {
      const text = document.getElementById("textFilter").value.toLowerCase();
      const group = document.getElementById("groupFilter").value;
      const tool = document.getElementById("toolFilter").value;
      const device = document.getElementById("deviceFilter").value;
      const workers = document.getElementById("workerFilter").value;
      const state = document.getElementById("stateFilter").value;
      return DATA.runs.filter(r => {
        const meta = r.meta || {};
        const rmeta = r.run_meta || {};
        if (group && r.group !== group) return false;
        if (tool && meta.tool !== tool) return false;
        if (device && meta.device !== device) return false;
        if (workers && String(meta.workers) !== workers) return false;
        if (state && jobState(r) !== state) return false;
        if (text) {
          const hay = (r.name + " " + (meta.tool || "") + " " + (rmeta.description || "")).toLowerCase();
          if (!hay.includes(text)) return false;
        }
        return true;
      });
    }

    function buildRunList(filtered) {
      const list = document.getElementById("runList");
      list.innerHTML = "";
      filtered.forEach((r, idx) => {
        const div = document.createElement("div");
        div.className = "run-item";
        const checked = "checked";
        const tags = [
          r.group ? "<span class=\\"tag\\">" + r.group + "</span>" : "",
          (r.meta||{}).device ? "<span class=\\"tag\\">" + r.meta.device + "</span>" : "",
          (r.meta||{}).workers ? "<span class=\\"tag\\">w" + r.meta.workers + "</span>" : "",
        ].join("");
        div.innerHTML = "<input type=\\"checkbox\\" data-run=\\"" + r.id + "\\" " + checked + ">" +
          "<div><strong>" + r.name + "</strong>" + tags + "<div class=\\"muted\\">" +
          ((r.run_meta||{}).description || "") + "</div></div>";
        list.appendChild(div);
      });
      list.querySelectorAll("input[type=checkbox]").forEach(cb => {
        cb.addEventListener("change", () => renderCharts(filtered));
      });
    }

    let chartExplore = null;
    let chartRate = null;
    let chartUpdate = null;
    let chartQuality = null;

    function selectedRuns(filtered) {
      const selectedIds = new Set();
      document.querySelectorAll("#runList input[type=checkbox]").forEach(cb => {
        if (cb.checked) selectedIds.add(cb.dataset.run);
      });
      return filtered.filter(r => selectedIds.has(r.id));
    }

    function buildLabels(runs, stageFilter) {
      const labels = new Set();
      runs.forEach(r => (r.points||[]).forEach(p => {
        if (!stageFilter || p.stage === stageFilter) labels.add(p.ts);
      }));
      return Array.from(labels).sort();
    }

    function renderRunDetails(runs) {
      const details = document.getElementById("runDetails");
      const sbatch = document.getElementById("runSbatch");
      const logLink = document.getElementById("runLogLink");
      const logPath = document.getElementById("runLogPath");
      const workerBox = document.getElementById("runWorkerLogs");
      if (runs.length !== 1) {
        details.textContent = runs.length ? "Select a single run to see full details." : "No runs selected.";
        sbatch.textContent = "";
        logLink.href = "#";
        logLink.style.pointerEvents = "none";
        logLink.style.opacity = "0.5";
        logPath.textContent = "";
        workerBox.textContent = "";
        return;
      }
      const run = runs[0];
      const meta = run.meta || {};
      const rmeta = run.run_meta || {};
      const job = jobMeta(run);
      const params = (run.csv || {}).params || {};
      const lines = [];
      lines.push("Run: " + run.name);
      lines.push("Group: " + run.group);
      if (rmeta.description) lines.push("Description: " + rmeta.description);
      if (meta.tool) lines.push("Tool: " + meta.tool);
      if (meta.device) lines.push("Device: " + meta.device);
      if (meta.workers) lines.push("Workers: " + meta.workers);
      if (job.State) lines.push("Job: " + job.State + " (" + (job.Elapsed || "?") + ")");
      if (job.NodeList) lines.push("Node: " + job.NodeList);
      if (job.Start) lines.push("Start: " + job.Start);
      if (job.End) lines.push("End: " + job.End);
      const s = run.summary || {};
      lines.push("Steps: heatup=" + (s.last_heatup || 0) + " explore=" + (s.last_explore || 0) + " update=" + (s.last_update || 0));
      if (Object.keys(params).length) {
        lines.push("");
        lines.push("Params:");
        for (const [k, v] of Object.entries(params)) {
          lines.push("  " + k + ": " + v);
        }
      }
      details.textContent = lines.join("\\n");
      sbatch.textContent = (rmeta.sbatch_text || "");
      logLink.href = run.log_href || "#";
      logLink.style.pointerEvents = "auto";
      logLink.style.opacity = "1";
      logPath.textContent = run.path || "";
      const workers = run.worker_logs || [];
      if (!workers.length) {
        workerBox.textContent = "No logs_subprocesses files found.";
      } else {
        workerBox.innerHTML = workers.map(w => "<div><a href=\\"" + w.href + "\\" target=\\"_blank\\">" + w.name + "</a></div>").join("");
      }
    }

    function renderCharts(filtered) {
      const runs = selectedRuns(filtered);
      if (!runs.length) return;
      const stageFilter = document.getElementById("stageFilter").value;
      const labels = buildLabels(runs, stageFilter);

      const datasetsExplore = runs.map((r, idx) => {
        const points = (r.points || []).filter(p => !stageFilter || p.stage === stageFilter);
        const map = new Map(points.map(p => [p.ts, p.explore]));
        const data = labels.map(ts => map.get(ts) ?? null);
        return {
          label: r.name,
          data,
          borderColor: COLORS[idx % COLORS.length],
          pointRadius: 0,
          tension: 0.2,
          spanGaps: true,
        };
      });

      const datasetsRate = runs.map((r, idx) => {
        const points = (r.points || []).filter(p => !stageFilter || p.stage === stageFilter);
        const map = new Map(points.map(p => [p.ts, p.rate]));
        const data = labels.map(ts => map.get(ts) ?? null);
        return {
          label: r.name,
          data,
          borderColor: COLORS[idx % COLORS.length],
          pointRadius: 0,
          tension: 0.2,
          spanGaps: true,
        };
      });

      const datasetsUpdate = runs.map((r, idx) => {
        const points = (r.points || []).filter(p => !stageFilter || p.stage === stageFilter);
        const map = new Map(points.map(p => [p.ts, p.update_rate]));
        const data = labels.map(ts => map.get(ts) ?? null);
        return {
          label: r.name,
          data,
          borderColor: COLORS[idx % COLORS.length],
          pointRadius: 0,
          tension: 0.2,
          spanGaps: true,
        };
      });

      const datasetsQuality = runs.map((r, idx) => {
        const evals = (r.csv || {}).evals || [];
        const data = evals
          .filter(e => e.steps !== null && e.quality !== null && e.quality !== undefined)
          .map(e => ({ x: e.steps, y: e.quality }));
        return {
          label: r.name,
          data,
          borderColor: COLORS[idx % COLORS.length],
          backgroundColor: COLORS[idx % COLORS.length],
          showLine: false,
          pointRadius: 3,
        };
      });

      const avgHeat = runs.reduce((acc, r) => acc + (((r.summary || {}).avg_heatup_rate) || 0), 0) / runs.length;
      const avgTrain = runs.reduce((acc, r) => acc + (((r.summary || {}).avg_train_rate) || 0), 0) / runs.length;
      const avgUpdate = runs.reduce((acc, r) => acc + (((r.summary || {}).avg_update_rate) || 0), 0) / runs.length;
      const tools = [...new Set(runs.map(r => (r.meta || {}).tool).filter(Boolean))];
      const devices = [...new Set(runs.map(r => (r.meta || {}).device).filter(Boolean))];
      const workers = [...new Set(runs.map(r => (r.meta || {}).workers).filter(Boolean))];
      document.getElementById("runMeta").textContent =
        "selected=" + runs.length +
        " | tools=" + (tools.join(",") || "?") +
        " devices=" + (devices.join(",") || "?") +
        " workers=" + (workers.join(",") || "?") +
        " | avg heatup/s=" + fmtNum(avgHeat, 2) +
        " train/s=" + fmtNum(avgTrain, 2) +
        " update/s=" + fmtNum(avgUpdate, 2);

      renderRunDetails(runs);

      if (chartExplore) chartExplore.destroy();
      if (chartRate) chartRate.destroy();
      if (chartUpdate) chartUpdate.destroy();
      if (chartQuality) chartQuality.destroy();

      chartExplore = new Chart(document.getElementById("chartExplore"), {
        type: "line",
        data: { labels, datasets: datasetsExplore },
        options: {
          responsive: true,
          scales: {
            x: { display: true },
            y: { display: true, title: { display: true, text: "Steps" } }
          }
        }
      });

      chartRate = new Chart(document.getElementById("chartRate"), {
        type: "line",
        data: { labels, datasets: datasetsRate },
        options: {
          responsive: true,
          scales: {
            x: { display: true },
            y: { display: true, title: { display: true, text: "Steps/s" } }
          }
        }
      });

      chartUpdate = new Chart(document.getElementById("chartUpdate"), {
        type: "line",
        data: { labels, datasets: datasetsUpdate },
        options: {
          responsive: true,
          scales: {
            x: { display: true },
            y: { display: true, title: { display: true, text: "Steps/s" } }
          }
        }
      });

      chartQuality = new Chart(document.getElementById("chartQuality"), {
        type: "scatter",
        data: { datasets: datasetsQuality },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: "Explore steps" } },
            y: { title: { display: true, text: "Quality" } }
          }
        }
      });
    }

    function refresh() {
      const filtered = applyFilters();
      buildSummary(filtered);
      buildRunList(filtered);
      renderCharts(filtered);
    }

    document.getElementById("textFilter").addEventListener("input", refresh);
    document.getElementById("groupFilter").addEventListener("change", refresh);
    document.getElementById("toolFilter").addEventListener("change", refresh);
    document.getElementById("deviceFilter").addEventListener("change", refresh);
    document.getElementById("workerFilter").addEventListener("change", refresh);
    document.getElementById("stateFilter").addEventListener("change", refresh);
    document.getElementById("stageFilter").addEventListener("change", () => renderCharts(applyFilters()));
    document.getElementById("selectAll").addEventListener("click", () => {
      document.querySelectorAll("#runList input[type=checkbox]").forEach(cb => cb.checked = true);
      refresh();
    });
    document.getElementById("clearSel").addEventListener("click", () => {
      document.querySelectorAll("#runList input[type=checkbox]").forEach(cb => cb.checked = false);
      refresh();
    });

    buildFilters();
    refresh();
  </script>
</body>
</html>
"""
    return template.replace("__PAYLOAD__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML dashboard from logs.")
    parser.add_argument("--input", type=str, default="logs/hpc/raw", help="Root folder with main.log files.")
    parser.add_argument("--out", type=str, default="logs/hpc/dashboard", help="Output folder.")
    args = parser.parse_args()

    input_root = Path(args.input)
    if not input_root.exists():
        raise SystemExit(f"Input path does not exist: {input_root}")

    out_root = Path(args.out)
    build_dashboard(input_root, out_root)


if __name__ == "__main__":
    main()
