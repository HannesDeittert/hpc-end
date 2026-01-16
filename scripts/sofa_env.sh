#!/usr/bin/env bash
# This script is meant to be sourced. Avoid leaking strict shell options into the
# caller shell (which can break e.g. `conda activate` under `set -u`).
__SOFA_ENV_SAVED_OPTS="$(set +o)"
set -euo pipefail

# Sets up SOFA/SofaPython3 for the current shell.
# Usage:
#   source scripts/sofa_env.sh
#
# Requirements:
# - Run inside the conda env (so $CONDA_PREFIX is set)
# - A SOFA binary install exists under $SOFA_ROOT (or one of the autodetected paths)

detect_sofa_root() {
  local candidates=(
    "${SOFA_ROOT:-}"
    "$HOME/opt/Sofa_v23.06.00_Linux/SOFA_v23.06.00_Linux"
    "$HOME/opt/SOFA_v23.06.00_Linux/SOFA_v23.06.00_Linux"
  )

  for c in "${candidates[@]}"; do
    if [[ -n "$c" && -d "$c/plugins/SofaPython3" ]]; then
      echo "$c"
      return 0
    fi
  done

  # Fallback: search under ~/opt for extracted binaries.
  local found
  found="$(find "$HOME/opt" -maxdepth 3 -type d -name 'SOFA_v*_Linux' 2>/dev/null | head -n 1 || true)"
  if [[ -n "$found" && -d "$found/plugins/SofaPython3" ]]; then
    echo "$found"
    return 0
  fi

  return 1
}

__sofa_env_main() {
  if [[ -z "${CONDA_PREFIX:-}" || ! -d "${CONDA_PREFIX:-}/lib" ]]; then
    echo "[sofa_env] error: conda env not active (CONDA_PREFIX missing)." >&2
    echo "[sofa_env] run: conda activate <env>  (must be Python 3.8)" >&2
    return 2
  fi

  SOFA_ROOT="$(detect_sofa_root)" || {
    echo "[sofa_env] error: could not detect SOFA_ROOT." >&2
    echo "[sofa_env] set it manually, e.g.:" >&2
    echo "  export SOFA_ROOT=\"$HOME/opt/Sofa_v23.06.00_Linux/SOFA_v23.06.00_Linux\"" >&2
    return 3
  }

  export SOFA_ROOT

  SOFA_PY="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages"
  if [[ -d "$SOFA_PY" ]]; then
    export PYTHONPATH="$SOFA_PY:${PYTHONPATH:-}"
  else
    echo "[sofa_env] warning: expected SofaPython3 site-packages not found at: $SOFA_PY" >&2
  fi

  # Needed so SofaPython3 can dlopen against the conda libpython (libpython3.8.so.1.0).
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  echo "[sofa_env] SOFA_ROOT=$SOFA_ROOT"
  echo "[sofa_env] PYTHONPATH prepended: $SOFA_PY"
  echo "[sofa_env] LD_LIBRARY_PATH prepended: $CONDA_PREFIX/lib"
}

__sofa_env_main "$@"
__sofa_env_status=$?

eval "$__SOFA_ENV_SAVED_OPTS"
unset __SOFA_ENV_SAVED_OPTS

return $__sofa_env_status 2>/dev/null || exit $__sofa_env_status
