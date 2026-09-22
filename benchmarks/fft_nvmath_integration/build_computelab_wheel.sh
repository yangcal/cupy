#!/usr/bin/env bash

# Build the CuPy nvmath-integration branch in node-local storage and retain
# only the resulting wheel on NFS.  This script performs no networked Git
# operations; the source checkout and its submodules must already be present.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WHEELHOUSE="${1:-${CUPY_INTEGRATION_WHEELHOUSE:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NVMATH_SPEC="${NVMATH_SPEC:-nvmath-python[cu13]==1.0.0}"
BUILD_JOBS="${CUPY_NUM_BUILD_JOBS:-16}"
NVCC_THREADS="${CUPY_NUM_NVCC_THREADS:-4}"

if [[ -z "$WHEELHOUSE" ]]; then
    echo "usage: $0 /nfs/path/to/wheelhouse" >&2
    exit 2
fi

for command in git tar "$PYTHON_BIN"; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "required command not found: $command" >&2
        exit 1
    fi
done

if git -C "$REPO_ROOT" submodule status --recursive | grep -Eq '^[-+U]'; then
    echo "CuPy submodules are missing or do not match the recorded commits." >&2
    echo "Initialize them on the frontend before requesting a compute node:" >&2
    echo "  git -C $REPO_ROOT submodule update --init --recursive" >&2
    exit 1
fi

CUPY_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
PYTHON_TAG="$($PYTHON_BIN -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
MACHINE="$(uname -m)"
WHEEL_KEY="${CUPY_COMMIT}-${PYTHON_TAG}-cu13-${MACHINE}"
DESTINATION="$WHEELHOUSE/$WHEEL_KEY"

mkdir -p "$DESTINATION"
EXISTING_WHEEL="$(find "$DESTINATION" -maxdepth 1 -type f -name 'cupy-*.whl' -print -quit)"
if [[ -n "$EXISTING_WHEEL" ]]; then
    echo "Reusing existing wheel: $EXISTING_WHEEL"
    exit 0
fi

NODE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cupy-integration-build.XXXXXX")"
VENV="$NODE_ROOT/venv"
LOCAL_SOURCE="$NODE_ROOT/cupy"
LOCAL_WHEELS="$NODE_ROOT/wheels"

echo "Node-local build root: $NODE_ROOT"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
    "$NVMATH_SPEC" \
    "cuda-toolkit[nvcc,profiler]==13.*" \
    "setuptools>=77" \
    wheel \
    "Cython>=3.2,<3.3,!=3.2.6" \
    "numpy>=2.0,<2.7"

CUDA_INCLUDE_DIR="$(python -c 'from cuda.pathfinder import find_nvidia_header_directory; print(find_nvidia_header_directory("cudart"))')"
CUDA_PATH="$(dirname "$CUDA_INCLUDE_DIR")"
NVCC="$(python -c 'from cuda.pathfinder import find_nvidia_binary_utility; print(find_nvidia_binary_utility("nvcc"))')"

if [[ ! -x "$NVCC" ]]; then
    echo "nvcc was not found in the node-local environment: $NVCC" >&2
    exit 1
fi

for header in \
    cuda.h \
    cuda_runtime.h \
    cuda_profiler_api.h \
    cublas_v2.h \
    cufft.h \
    curand.h \
    cusparse.h
do
    if [[ ! -f "$CUDA_INCLUDE_DIR/$header" ]]; then
        echo "required CUDA header not found: $CUDA_INCLUDE_DIR/$header" >&2
        exit 1
    fi
done

link_shared_library() {
    local stem="$1"
    local target
    target="$(find "$CUDA_PATH/lib" -maxdepth 1 -type f -name "${stem}.so.*" -printf '%f\n' | sort -V | tail -n 1)"
    if [[ -z "$target" ]]; then
        echo "required CUDA library not found: ${stem}.so.*" >&2
        exit 1
    fi
    ln -sfn "$target" "$CUDA_PATH/lib/${stem}.so"
}

for library in \
    libcudart \
    libcublas \
    libcufft \
    libcurand \
    libcusparse \
    libcusolver \
    libnvrtc \
    libnvJitLink
do
    link_shared_library "$library"
done

export CUDA_PATH
export NVCC
export PATH="$(dirname "$NVCC"):$PATH"
export CFLAGS="-I$CUDA_INCLUDE_DIR"
export CXXFLAGS="-I$CUDA_INCLUDE_DIR"
export CPATH="$CUDA_INCLUDE_DIR${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_PATH/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LDFLAGS="-L$CUDA_PATH/lib -Wl,-rpath,$CUDA_PATH/lib"
export CUPY_USE_CUDA_PYTHON=1
export CUPY_NUM_BUILD_JOBS="$BUILD_JOBS"
export CUPY_NUM_NVCC_THREADS="$NVCC_THREADS"

mkdir -p "$LOCAL_SOURCE" "$LOCAL_WHEELS"
tar \
    -C "$REPO_ROOT" \
    --exclude-vcs \
    --exclude=build \
    -cf - . \
    | tar -C "$LOCAL_SOURCE" -xf -

if [[ -z "$(find "$LOCAL_SOURCE/third_party/cccl" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "the node-local CuPy copy has an empty CCCL submodule" >&2
    exit 1
fi

echo "Building CuPy commit $CUPY_COMMIT with $BUILD_JOBS jobs"
cd "$LOCAL_SOURCE"
python -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "$LOCAL_WHEELS" \
    .

BUILT_WHEEL="$(find "$LOCAL_WHEELS" -maxdepth 1 -type f -name 'cupy-*.whl' -print -quit)"
if [[ -z "$BUILT_WHEEL" ]]; then
    echo "CuPy build completed without producing a wheel" >&2
    exit 1
fi

FINAL_WHEEL="$DESTINATION/$(basename "$BUILT_WHEEL")"
cp "$BUILT_WHEEL" "$FINAL_WHEEL"

{
    printf 'cupy_commit=%s\n' "$CUPY_COMMIT"
    printf 'python_tag=%s\n' "$PYTHON_TAG"
    printf 'machine=%s\n' "$MACHINE"
    printf 'nvmath_spec=%s\n' "$NVMATH_SPEC"
    printf 'cuda_path=%s\n' "$CUDA_PATH"
    printf 'nvcc=%s\n' "$NVCC"
    printf 'build_jobs=%s\n' "$BUILD_JOBS"
    printf 'nvcc_threads=%s\n' "$NVCC_THREADS"
    python -m pip freeze
} > "$DESTINATION/build-manifest.txt"

sha256sum "$FINAL_WHEEL" > "$DESTINATION/sha256.txt"

echo "Custom CuPy wheel: $FINAL_WHEEL"
echo "Build manifest: $DESTINATION/build-manifest.txt"

