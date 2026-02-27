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
            points.append(
                {
                    "ts": ts,
                    "stage": m.group("stage"),
                    "heatup": int(m.group("heatup")),
                    "explore": int(m.group("explore")),
                    "update": int(m.group("update")),
                    "dheatup": int(m.group("dheatup")),
                    "dexplore": int(m.group("dexplore")),
                    "dupdate": int(m.group("dupdate")),
                    "dt": int(m.group("dt")),
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


def _summarize(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not points:
        return {
            "start_ts": None,
            "end_ts": None,
            "duration_s": None,
            "last_heatup": 0,
            "last_explore": 0,
            "last_update": 0,
            "avg_explore_rate": 0.0,
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
    rates = []
    for p in points:
        if p["dt"] > 0 and p["dexplore"] > 0:
            rates.append(p["dexplore"] / p["dt"])
    avg_rate = sum(rates) / len(rates) if rates else 0.0

    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": duration_s,
        "last_heatup": last["heatup"],
        "last_explore": last["explore"],
        "last_update": last["update"],
        "avg_explore_rate": round(avg_rate, 3),
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
        summary = _summarize(points)
        run = {
            "id": str(run_dir),
            "name": run_dir.name,
            "group": _guess_group(log),
            "path": str(log),
            "meta": meta,
            "run_meta": run_meta,
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
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HPC Training Dashboard</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Arial; margin: 24px; }}
    .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f6f6; }}
    canvas {{ max-width: 100%; }}
    .muted {{ color: #666; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>HPC Training Dashboard</h1>
  <p class="muted">Auto-generated from main.log heartbeats.</p>

  <div class="card">
    <div style="display:flex; gap:12px; flex-wrap: wrap; align-items: center;">
      <label>Filter:</label>
      <input id="textFilter" type="text" placeholder="search name/tool/desc" />
      <select id="groupFilter"></select>
      <select id="toolFilter"></select>
      <select id="deviceFilter"></select>
      <select id="workerFilter"></select>
      <button id="selectAll">Select All</button>
      <button id="clearSel">Clear</button>
    </div>
    <div id="runMeta" class="muted" style="margin-top:8px;"></div>
    <div id="runList" style="margin-top:8px;"></div>
  </div>

  <div class="row" style="margin-top: 16px;">
    <div class="card" style="flex: 1 1 480px;">
      <h3>Explore Steps vs Time</h3>
      <canvas id="chartExplore"></canvas>
    </div>
    <div class="card" style="flex: 1 1 480px;">
      <h3>Explore Steps/s (per heartbeat)</h3>
      <canvas id="chartRate"></canvas>
    </div>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h3>Runs Summary</h3>
    <table id="summaryTable">
      <thead>
        <tr>
          <th>Run</th>
          <th>Group</th>
          <th>Tool</th>
          <th>Device</th>
          <th>Workers</th>
          <th>Description</th>
          <th>Start</th>
          <th>End</th>
          <th>Explore</th>
          <th>Update</th>
          <th>Avg Explore/s</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <script>
    const DATA = {payload};

    function fmt(val) {{
      return (val === null || val === undefined) ? "" : String(val);
    }}

    function buildSummary(filtered) {{
      const tbody = document.querySelector("#summaryTable tbody");
      tbody.innerHTML = "";
      for (const run of filtered) {{
        const tr = document.createElement("tr");
        const meta = run.meta || {{}};
        const rmeta = run.run_meta || {{}};
        const s = run.summary || {{}};
        const cells = [
          run.name,
          run.group,
          meta.tool || "",
          meta.device || "",
          meta.workers || "",
          rmeta.description || "",
          s.start_ts || "",
          s.end_ts || "",
          s.last_explore || 0,
          s.last_update || 0,
          s.avg_explore_rate || 0,
        ];
        for (const c of cells) {{
          const td = document.createElement("td");
          td.textContent = fmt(c);
          tr.appendChild(td);
        }}
        tbody.appendChild(tr);
      }}
    }}

    function uniqueValues(keyFn, runs) {{
      const vals = new Set();
      runs.forEach(r => {{
        const v = keyFn(r);
        if (v !== undefined && v !== null && v !== "") vals.add(v);
      }});
      return Array.from(vals).sort();
    }}

    function setSelectOptions(sel, values, label) {{
      sel.innerHTML = "";
      const all = document.createElement("option");
      all.value = "";
      all.textContent = `All ${label}`;
      sel.appendChild(all);
      values.forEach(v => {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      }});
    }}

    function buildFilters() {{
      setSelectOptions(document.getElementById("groupFilter"),
        uniqueValues(r => r.group, DATA.runs), "groups");
      setSelectOptions(document.getElementById("toolFilter"),
        uniqueValues(r => (r.meta||{{}}).tool, DATA.runs), "tools");
      setSelectOptions(document.getElementById("deviceFilter"),
        uniqueValues(r => (r.meta||{{}}).device, DATA.runs), "devices");
      setSelectOptions(document.getElementById("workerFilter"),
        uniqueValues(r => (r.meta||{{}}).workers, DATA.runs).map(String), "workers");
    }}

    function applyFilters() {{
      const text = document.getElementById("textFilter").value.toLowerCase();
      const group = document.getElementById("groupFilter").value;
      const tool = document.getElementById("toolFilter").value;
      const device = document.getElementById("deviceFilter").value;
      const workers = document.getElementById("workerFilter").value;
      return DATA.runs.filter(r => {{
        const meta = r.meta || {{}};
        const rmeta = r.run_meta || {{}};
        if (group && r.group !== group) return false;
        if (tool && meta.tool !== tool) return false;
        if (device && meta.device !== device) return false;
        if (workers && String(meta.workers) !== workers) return false;
        if (text) {{
          const hay = `${{r.name}} ${{meta.tool||""}} ${{rmeta.description||""}}`.toLowerCase();
          if (!hay.includes(text)) return false;
        }}
        return true;
      }});
    }}

    function buildRunList(filtered) {{
      const list = document.getElementById("runList");
      list.innerHTML = "";
      filtered.forEach((r, idx) => {{
        const id = `run_${{idx}}`;
        const div = document.createElement("div");
        div.innerHTML = `<label><input type="checkbox" data-run="${{r.id}}" checked> ${{r.name}}</label>`;
        list.appendChild(div);
      }});
      list.querySelectorAll("input[type=checkbox]").forEach(cb => {{
        cb.addEventListener("change", () => renderCharts(filtered));
      }});
    }}

    let chartExplore = null;
    let chartRate = null;

    function selectedRuns(filtered) {{
      const selectedIds = new Set();
      document.querySelectorAll("#runList input[type=checkbox]").forEach(cb => {{
        if (cb.checked) selectedIds.add(cb.dataset.run);
      }});
      return filtered.filter(r => selectedIds.has(r.id));
    }}

    function buildLabels(runs) {{
      const labels = new Set();
      runs.forEach(r => (r.points||[]).forEach(p => labels.add(p.ts)));
      return Array.from(labels).sort();
    }}

    function renderCharts(filtered) {{
      const runs = selectedRuns(filtered);
      if (!runs.length) return;
      const labels = buildLabels(runs);

      const datasetsExplore = runs.map((r, idx) => {{
        const map = new Map((r.points||[]).map(p => [p.ts, p.explore]));
        const data = labels.map(ts => map.get(ts) ?? null);
        return {{
          label: r.name,
          data,
          borderColor: ["#2563eb","#16a34a","#f97316","#a855f7","#ef4444"][idx % 5],
          pointRadius: 0,
        }};
      }});

      const datasetsRate = runs.map((r, idx) => {{
        const map = new Map((r.points||[]).map(p => [p.ts, (p.dt && p.dexplore) ? (p.dexplore / p.dt) : 0]));
        const data = labels.map(ts => map.get(ts) ?? null);
        return {{
          label: r.name,
          data,
          borderColor: ["#0ea5e9","#22c55e","#f59e0b","#8b5cf6","#ef4444"][idx % 5],
          pointRadius: 0,
        }};
      }});

      const meta = runs[0].meta || {{}};
      const s = runs[0].summary || {{}};
      document.getElementById("runMeta").textContent =
        `selected=${runs.length} | tool=${meta.tool || "?"} device=${meta.device || "?"} workers=${meta.workers || "?"} ` +
        `| explore=${s.last_explore || 0} update=${s.last_update || 0} avg_explore/s=${s.avg_explore_rate || 0}`;

      if (chartExplore) chartExplore.destroy();
      if (chartRate) chartRate.destroy();

      chartExplore = new Chart(document.getElementById("chartExplore"), {{
        type: "line",
        data: {{ labels, datasets: datasetsExplore }},
        options: {{
          responsive: true,
          scales: {{
            x: {{ display: true }},
            y: {{ display: true, title: {{ display: true, text: "Steps" }} }}
          }}
        }}
      }});

      chartRate = new Chart(document.getElementById("chartRate"), {{
        type: "line",
        data: {{ labels, datasets: datasetsRate }},
        options: {{
          responsive: true,
          scales: {{
            x: {{ display: true }},
            y: {{ display: true, title: {{ display: true, text: "Steps/s" }} }}
          }}
        }}
      }});
    }}

    function refresh() {{
      const filtered = applyFilters();
      buildSummary(filtered);
      buildRunList(filtered);
      renderCharts(filtered);
    }}

    document.getElementById("textFilter").addEventListener("input", refresh);
    document.getElementById("groupFilter").addEventListener("change", refresh);
    document.getElementById("toolFilter").addEventListener("change", refresh);
    document.getElementById("deviceFilter").addEventListener("change", refresh);
    document.getElementById("workerFilter").addEventListener("change", refresh);
    document.getElementById("selectAll").addEventListener("click", () => {{
      document.querySelectorAll("#runList input[type=checkbox]").forEach(cb => cb.checked = true);
      refresh();
    }});
    document.getElementById("clearSel").addEventListener("click", () => {{
      document.querySelectorAll("#runList input[type=checkbox]").forEach(cb => cb.checked = false);
      refresh();
    }});

    buildFilters();
    refresh();
  </script>
</body>
</html>
"""


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
