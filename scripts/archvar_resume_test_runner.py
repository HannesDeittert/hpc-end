#!/usr/bin/env python3
import argparse
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


DEFAULT_RESULTS = Path(
    "/home/woody/iwhr/iwhr106h/master-project/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94"
)


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("resume_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.CONFIG


def checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint(\d+)\.everl$", path.name)
    return int(match.group(1)) if match else -1


def load_ckpt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_old_run(results: Path, original_run: str) -> Path:
    run_dir = results / original_run
    if run_dir.is_dir():
        return run_dir

    matches = sorted(p for p in results.glob(f"*{original_run}*") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find old run folder: {original_run}")
    return matches[-1]


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = list((run_dir / "checkpoints").glob("checkpoint*.everl"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint*.everl found in {run_dir / 'checkpoints'}")
    return sorted(ckpts, key=checkpoint_number)[-1]


def tool_preflight(module_name: str, class_name: str):
    from steve_recommender.bench import resolve_device, build_archvar_intervention

    device, label = resolve_device("unused", module_name, class_name)
    intervention = build_archvar_intervention(device=device)

    print("label:", label)
    print("device:", device.__class__.__name__)
    print("simulation:", type(intervention.simulation).__name__)
    print("fluoroscopy:", type(intervention.fluoroscopy).__name__)


def checkpoint_preflight(ckpt_path: Path, sidecar_path: Path):
    ckpt = load_ckpt(ckpt_path)
    side = load_ckpt(sidecar_path)

    print("checkpoint:", ckpt_path)
    print("sidecar:", sidecar_path)
    print("ckpt steps:", ckpt.get("steps"))
    print("side steps:", side.get("steps"))
    print("ckpt episodes:", ckpt.get("episodes"))
    print("side episodes:", side.get("episodes"))

    if ckpt.get("steps") != side.get("steps"):
        raise RuntimeError("Checkpoint and sidecar steps do not match")

    if ckpt.get("episodes") != side.get("episodes"):
        raise RuntimeError("Checkpoint and sidecar episodes do not match")

    print("OK: checkpoint and sidecar match")


def find_new_run_dir(results: Path, run_name: str):
    matches = sorted(p for p in results.glob(f"*{run_name}*") if p.is_dir())
    return matches[-1] if matches else None


def terminate_process_group(proc: subprocess.Popen):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--nw", type=int, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-min", type=int, default=45)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--delete-on-success", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = args.results

    token = args.tag or os.environ.get("SLURM_JOB_ID") or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_name = f"{cfg['run_prefix']}_test_nw{args.nw}_{token}"

    print("=== config ===", flush=True)
    print("name:", cfg["name"], flush=True)
    print("original_run:", cfg["original_run"], flush=True)
    print("module:", cfg["module"], flush=True)
    print("class:", cfg["class"], flush=True)
    print("run_prefix:", cfg["run_prefix"], flush=True)
    print("run_name:", run_name, flush=True)
    print("results:", results, flush=True)
    print("nw:", args.nw, flush=True)
    print("device:", args.device, flush=True)

    old_run_dir = find_old_run(results, cfg["original_run"])
    ckpt = latest_checkpoint(old_run_dir)
    sidecar = old_run_dir / "checkpoints" / "latest_replay_buffer.everl"

    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing sidecar replay buffer: {sidecar}")

    print("\n=== paths ===", flush=True)
    print("old_run_dir:", old_run_dir, flush=True)
    print("ckpt:", ckpt, flush=True)
    print("sidecar:", sidecar, flush=True)

    print("\n=== tool preflight ===", flush=True)
    tool_preflight(cfg["module"], cfg["class"])

    print("\n=== checkpoint preflight ===", flush=True)
    checkpoint_preflight(ckpt, sidecar)

    cmd = [
        sys.executable,
        "scripts/archvar_train_tool.py",
        "-d", args.device,
        "-nw", str(args.nw),
        "-n", run_name,
        "--heatup-steps", "500000",
        "--training-steps", "20000000",
        "--explore-steps-between-eval", "250000",
        "--explore-episodes-between-updates", "100",
        "-lr", "0.0003218",
        "--hidden", "400", "400", "400",
        "-en", "900",
        "-el", "1",
        "--train-max-steps", "150",
        "--eval-max-steps", "450",
        "--save-latest-replay-buffer",
        "--resume-from", str(ckpt),
        "--resume-replay-buffer-from", str(sidecar),
        "--tool-module", cfg["module"],
        "--tool-class", cfg["class"],
        "--results-folder", str(results),
    ]

    print("\n=== command ===", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd, cwd=Path.cwd(), start_new_session=True)

    deadline = time.time() + args.timeout_min * 60
    new_run_dir = None
    already_recorded = False

    print("\n=== monitoring until first update / exploration ===", flush=True)

    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"ERROR: process exited early with code {proc.returncode}", flush=True)
            return proc.returncode or 1

        if new_run_dir is None:
            new_run_dir = find_new_run_dir(results, run_name)

        if new_run_dir is not None and not already_recorded:
            print("TEST_RUN_DIR:", new_run_dir, flush=True)
            with open("scripts/.archvar_resume_test_runs.txt", "a") as f:
                f.write(str(new_run_dir) + "\n")
            already_recorded = True

        if new_run_dir is not None:
            main_log = new_run_dir / "main.log"
            trainer_log = new_run_dir / "logs_subprocesses" / "trainer_synchron.log"

            main_text = read_text(main_log)
            trainer_text = read_text(trainer_log)
            combined = main_text + "\n" + trainer_text

            if "Traceback" in combined or "RuntimeError" in combined or "Segmentation" in combined:
                print("WARNING: error-looking text found; inspect logs if test stalls.", flush=True)

            if "update / exploration" in main_text:
                print("\nSUCCESS: first update / exploration reached", flush=True)
                print("TEST_RUN_DIR:", new_run_dir, flush=True)

                print("\n--- last main.log lines ---", flush=True)
                for line in main_text.splitlines()[-20:]:
                    print(line, flush=True)

                terminate_process_group(proc)

                if args.delete_on_success:
                    csv = Path(str(new_run_dir) + ".csv")
                    print("Deleting:", new_run_dir, flush=True)
                    subprocess.run(["rm", "-rf", "--", str(new_run_dir)], check=False)
                    if csv.exists():
                        print("Deleting:", csv, flush=True)
                        csv.unlink()

                return 0

        time.sleep(10)

    print("\nERROR: timeout waiting for update / exploration", flush=True)
    print("new_run_dir:", new_run_dir, flush=True)
    terminate_process_group(proc)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())