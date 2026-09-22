#!/usr/bin/env bash

# Create a disposable node-local runtime environment, install the persisted
# custom CuPy wheel, and collect the hot-cache and cold-cache experiments.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CUSTOM_CUPY_WHEEL="${1:-}"
ARTIFACT_ROOT="${2:-${CUPY_INTEGRATION_ARTIFACT_ROOT:-/home/scratch.ajdesai_ent/artifacts/cupy/fft-nvmath-integration}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NVMATH_SPEC="${NVMATH_SPEC:-nvmath-python[cu13]==1.0.0}"

if [[ -z "$CUSTOM_CUPY_WHEEL" ]]; then
    echo "usage: $0 /nfs/path/to/custom-cupy.whl [/nfs/artifact/root]" >&2
    exit 2
fi
if [[ ! -f "$CUSTOM_CUPY_WHEEL" ]]; then
    echo "custom CuPy wheel not found: $CUSTOM_CUPY_WHEEL" >&2
    exit 1
fi

CUPY_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
WHEEL_DIRECTORY="$(dirname "$CUSTOM_CUPY_WHEEL")"
BUILD_MANIFEST="$WHEEL_DIRECTORY/build-manifest.txt"
if [[ ! -f "$BUILD_MANIFEST" ]]; then
    echo "custom CuPy build manifest not found: $BUILD_MANIFEST" >&2
    exit 1
fi
WHEEL_COMMIT="$(sed -n 's/^cupy_commit=//p' "$BUILD_MANIFEST")"
if [[ "$WHEEL_COMMIT" != "$CUPY_COMMIT" ]]; then
    echo "the custom wheel does not match the benchmark checkout" >&2
    echo "wheel commit: $WHEEL_COMMIT" >&2
    echo "source commit: $CUPY_COMMIT" >&2
    exit 1
fi
if [[ -f "$WHEEL_DIRECTORY/sha256.txt" ]]; then
    sha256sum --check "$WHEEL_DIRECTORY/sha256.txt"
fi

NODE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cupy-integration-run.XXXXXX")"
VENV="$NODE_ROOT/venv"

echo "Node-local runtime root: $NODE_ROOT"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install "$NVMATH_SPEC" matplotlib
python -m pip install --no-deps "$CUSTOM_CUPY_WHEEL"

CUDA_INCLUDE_DIR="$(python -c 'from cuda.pathfinder import find_nvidia_header_directory; print(find_nvidia_header_directory("cudart"))')"
CUDA_PATH="$(dirname "$CUDA_INCLUDE_DIR")"
export LD_LIBRARY_PATH="$CUDA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$NODE_ROOT"
python - <<'PY'
from importlib.metadata import version

import cupy as cp
import nvmath

print("CuPy:", cp.__file__)
print("CuPy version:", cp.__version__)
print("nvmath:", nvmath.__file__)
print("nvmath version:", version("nvmath-python"))
print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])
print("nvmath backend:", hasattr(cp.fft.config, "use_nvmath"))

assert hasattr(cp.fft.config, "use_nvmath")
PY

CUPY_SHORT="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)-${CUPY_SHORT}"
RUN_DIR="$ARTIFACT_ROOT/$RUN_TAG"

mkdir -p \
    "$RUN_DIR/metadata" \
    "$RUN_DIR/smoke" \
    "$RUN_DIR/hot" \
    "$RUN_DIR/cold"

export MPLCONFIGDIR="$NODE_ROOT/matplotlib"

printf '%s\n' "$CUPY_COMMIT" > "$RUN_DIR/metadata/cupy-commit.txt"
printf '%s\n' "$CUSTOM_CUPY_WHEEL" > "$RUN_DIR/metadata/cupy-wheel.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/metadata/cupy-status.txt"
python -m pip freeze > "$RUN_DIR/metadata/pip-freeze.txt"
nvidia-smi -q > "$RUN_DIR/metadata/nvidia-smi.txt"
hostname > "$RUN_DIR/metadata/hostname.txt"

python "$SCRIPT_DIR/run.py" \
    --mode all \
    --suite smoke \
    --hot-warmup 2 \
    --hot-iters 3 \
    --hot-rounds 2 \
    --hot-completion-iters 2 \
    --hot-completion-rounds 2 \
    --cold-warmup 2 \
    --cold-iters 2 \
    --cold-rounds 2 \
    --json "$RUN_DIR/smoke/raw.json" \
    2>&1 | tee "$RUN_DIR/smoke/collection.txt"

python "$SCRIPT_DIR/plot.py" \
    --json "$RUN_DIR/smoke/raw.json" \
    --output-dir "$RUN_DIR/smoke" \
    2>&1 | tee "$RUN_DIR/smoke/report.txt"

python "$SCRIPT_DIR/run.py" \
    --mode hot \
    --fixture c2c_len1024_c64 \
    --json "$RUN_DIR/hot/raw.json" \
    2>&1 | tee "$RUN_DIR/hot/collection.txt"

python "$SCRIPT_DIR/plot.py" \
    --json "$RUN_DIR/hot/raw.json" \
    --output-dir "$RUN_DIR/hot" \
    2>&1 | tee "$RUN_DIR/hot/report.txt"

python "$SCRIPT_DIR/run.py" \
    --mode cold \
    --suite full \
    --json "$RUN_DIR/cold/raw.json" \
    2>&1 | tee "$RUN_DIR/cold/collection.txt"

python "$SCRIPT_DIR/plot.py" \
    --json "$RUN_DIR/cold/raw.json" \
    --output-dir "$RUN_DIR/cold" \
    2>&1 | tee "$RUN_DIR/cold/report.txt"

find "$RUN_DIR" -maxdepth 3 -type f -printf '%P\n' | sort
echo "Artifact directory: $RUN_DIR"
