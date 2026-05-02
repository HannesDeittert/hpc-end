#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPU_CORES="$("${PYTHON_BIN}" - <<'PY'
import os
print(len(os.sched_getaffinity(0)))
PY
)"
DEFAULT_WORKER_COUNT="$(( CPU_CORES > 5 ? CPU_CORES - 5 : 1 ))"

SAMPLE_JSON="${SAMPLE_JSON:-${PROJECT_ROOT}/results/experimental_prep/sample_12.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/master_thesis/e0}"
TRIAL_COUNT="${TRIAL_COUNT:-100}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
WORKER_COUNT="${WORKER_COUNT:-${DEFAULT_WORKER_COUNT}}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
BASE_SEED_START="${BASE_SEED_START:-123}"
TARGET_SEED_START="${TARGET_SEED_START:-9000}"
BASE_SEED_STEP="${BASE_SEED_STEP:-200}"
RUNS_PER_ANATOMY="${RUNS_PER_ANATOMY:-3}"
THRESHOLD_MM="${THRESHOLD_MM:-5.0}"

WIRES=(
  "steve_default/tight_j"
  "steve_default/standard_j"
  "steve_default/gentle"
  "amplatz_super_stiff/tight_j"
  "amplatz_super_stiff/standard_j"
  "amplatz_super_stiff/gentle"
  "universal_ii/tight_j"
  "universal_ii/standard_j"
  "universal_ii/gentle"
)

mapfile -t ANATOMIES < <(
  "${PYTHON_BIN}" - "${SAMPLE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

sample_path = Path(sys.argv[1])
payload = json.loads(sample_path.read_text(encoding="utf-8"))
selected = payload.get("selected_anatomies", [])
for item in selected:
    print(item["record_id"])
PY
)

if [ "${#ANATOMIES[@]}" -eq 0 ]; then
  echo "[E0] no anatomies found in ${SAMPLE_JSON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[E0] cpu_cores=${CPU_CORES} worker_count=${WORKER_COUNT} default_worker_count=${DEFAULT_WORKER_COUNT}"

base_seed="${BASE_SEED_START}"
target_seed="${TARGET_SEED_START}"
run_index=0
manifest_path="${OUTPUT_ROOT}/commands.json"
tmp_manifest="$(mktemp)"

"${PYTHON_BIN}" - "${tmp_manifest}" "${SAMPLE_JSON}" "${OUTPUT_ROOT}" "${BASE_SEED_START}" "${TARGET_SEED_START}" "${BASE_SEED_STEP}" "${RUNS_PER_ANATOMY}" "${TRIAL_COUNT}" "${WORKER_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
sample_json = sys.argv[2]
output_root = sys.argv[3]
base_seed_start = int(sys.argv[4])
target_seed_start = int(sys.argv[5])
base_seed_step = int(sys.argv[6])
runs_per_anatomy = int(sys.argv[7])
trial_count = int(sys.argv[8])
worker_count = int(sys.argv[9])

payload = {
    "sample_json": sample_json,
    "output_root": output_root,
    "base_seed_start": base_seed_start,
    "target_seed_start": target_seed_start,
    "base_seed_step": base_seed_step,
    "runs_per_anatomy": runs_per_anatomy,
    "trial_count": trial_count,
    "worker_count": worker_count,
    "execution_wires": [],
    "commands": [],
}
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

append_manifest_entry() {
  local entry="$1"
  "${PYTHON_BIN}" - "${tmp_manifest}" "${entry}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
entry = json.loads(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
payload.setdefault("commands", []).append(entry)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

for anatomy in "${ANATOMIES[@]}"; do
  for repeat_index in $(seq 0 $((RUNS_PER_ANATOMY - 1))); do
    job_name="${anatomy}_target_${target_seed}_seed_${base_seed}"
    scenario_name="${anatomy}_run_${run_index}"

    cmd=(
      "${PYTHON_BIN}" -m steve_recommender.eval_v2.cli run
      --job-name "${job_name}"
      --scenario-name "${scenario_name}"
      --anatomy "${anatomy}"
      --target-mode centerline_random
      --target-seed "${target_seed}"
      --trial-count "${TRIAL_COUNT}"
      --base-seed "${base_seed}"
      --policy-mode deterministic
      --max-episode-steps "${MAX_EPISODE_STEPS}"
      --workers "${WORKER_COUNT}"
      --policy-device "${POLICY_DEVICE}"
      --output-root "${OUTPUT_ROOT}"
      --threshold-mm "${THRESHOLD_MM}"
    )

    for wire in "${WIRES[@]}"; do
      cmd+=(--execution-wire "${wire}")
    done

    echo "[E0] ${cmd[*]}"
    "${cmd[@]}"

    entry="$("${PYTHON_BIN}" - "${anatomy}" "${repeat_index}" "${run_index}" "${base_seed}" "${target_seed}" "${job_name}" "${scenario_name}" <<'PY'
import json
import sys

anatomy, repeat_index, run_index, base_seed, target_seed, job_name, scenario_name = sys.argv[1:8]
print(json.dumps({
    "anatomy": anatomy,
    "repeat_index": int(repeat_index),
    "run_index": int(run_index),
    "base_seed": int(base_seed),
    "target_seed": int(target_seed),
    "job_name": job_name,
    "scenario_name": scenario_name,
    "execution_wires": [
        "steve_default/tight_j",
        "steve_default/standard_j",
        "steve_default/gentle",
        "amplatz_super_stiff/tight_j",
        "amplatz_super_stiff/standard_j",
        "amplatz_super_stiff/gentle",
        "universal_ii/tight_j",
        "universal_ii/standard_j",
        "universal_ii/gentle",
    ],
}, sort_keys=True))
PY
)"
    append_manifest_entry "${entry}"

    base_seed=$((base_seed + BASE_SEED_STEP))
    target_seed=$((target_seed + 1))
    run_index=$((run_index + 1))
  done
done

python_cmd="${PYTHON_BIN}"
"${python_cmd}" - "${tmp_manifest}" "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
PY

rm -f "${tmp_manifest}"
echo "[E0] wrote manifest to ${manifest_path}"
