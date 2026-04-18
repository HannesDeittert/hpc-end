#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${SOFA_ROOT:-}" ]]; then
  if [[ -d "$HOME/opt/Sofa_v23.06.00_Linux/SOFA_v23.06.00_Linux" ]]; then
    SOFA_ROOT="$HOME/opt/Sofa_v23.06.00_Linux/SOFA_v23.06.00_Linux"
  elif [[ -d "/data/H_deittert/opt/SOFA_v23.06.00_Linux" ]]; then
    SOFA_ROOT="/data/H_deittert/opt/SOFA_v23.06.00_Linux"
  elif [[ -d "/data/H_deittert/opt/SOFA_v23.06.00_Linux/SOFA_v23.06.00_Linux" ]]; then
    SOFA_ROOT="/data/H_deittert/opt/SOFA_v23.06.00_Linux/SOFA_v23.06.00_Linux"
  else
    echo "[wall-force] SOFA_ROOT is not set and no default install was found." >&2
    echo "[wall-force] Export SOFA_ROOT first, e.g.:" >&2
    echo "  export SOFA_ROOT=/path/to/SOFA_v23.06.00_Linux" >&2
    exit 1
  fi
fi

BUILD_DIR="$ROOT_DIR/native/sofa_wire_force_monitor/build"
LIB_PATH="$BUILD_DIR/libSofaWireForceMonitor.so"
CMAKE_ARGS=()
if [[ -x /usr/bin/cmake ]]; then
  CMAKE_BIN=/usr/bin/cmake
else
  CMAKE_BIN="$(command -v cmake)"
fi

if [[ -n "${BOOST_INCLUDEDIR:-}" ]]; then
  CMAKE_ARGS+=("-DBoost_INCLUDE_DIR=$BOOST_INCLUDEDIR")
elif [[ -d /usr/include/boost ]]; then
  CMAKE_ARGS+=("-DBoost_INCLUDE_DIR=/usr/include")
else
  cat <<'MSG' >&2
[wall-force] warning: Boost headers not found at /usr/include/boost.
[wall-force] if CMake fails with "Could NOT find Boost", install boost headers
[wall-force] or export BOOST_INCLUDEDIR=/path/to/boost/include.
MSG
fi

if [[ -n "${EIGEN3_INCLUDE_DIR:-}" ]]; then
  CMAKE_ARGS+=("-DEIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR")
elif [[ -d /usr/include/eigen3 ]]; then
  CMAKE_ARGS+=("-DEIGEN3_INCLUDE_DIR=/usr/include/eigen3")
else
  cat <<'MSG' >&2
[wall-force] warning: Eigen3 headers not found at /usr/include/eigen3.
[wall-force] if CMake fails with "Could NOT find Eigen3", install:
[wall-force]   sudo apt install libeigen3-dev
[wall-force] or export EIGEN3_INCLUDE_DIR=/path/to/eigen3.
MSG
fi

# Conda often injects libstdc++/libtinfo via LD_LIBRARY_PATH which can break
# system cmake ("GLIBCXX_x.y.z not found"). Run cmake with LD_LIBRARY_PATH unset.
env -u LD_LIBRARY_PATH "$CMAKE_BIN" \
  -S "$ROOT_DIR/native/sofa_wire_force_monitor" \
  -B "$BUILD_DIR" \
  -DSOFA_ROOT="$SOFA_ROOT" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  "${CMAKE_ARGS[@]}"

env -u LD_LIBRARY_PATH "$CMAKE_BIN" --build "$BUILD_DIR" --parallel "$(nproc)"

if [[ ! -f "$LIB_PATH" ]]; then
  echo "[wall-force] Build completed but library not found: $LIB_PATH" >&2
  exit 1
fi

cat <<MSG
[wall-force] build ok: $LIB_PATH
[wall-force] export this in your shell before evaluation/comparison:
  export STEVE_WALL_FORCE_MONITOR_PLUGIN="$LIB_PATH"
MSG
