#!/usr/bin/env bash
# Build the standalone CALVIN benchmark node (thin freeze).
#
# Frozen: node code + dora-rs / forge-msgs / forge-tool / numpy (pinned to the
#         host version) / pyarrow / pydantic / pyyaml / pillow.
# Host-provided at runtime (injected by adapters/calvin_adapter.py):
#         calvin_env / calvin_models / pybullet / torch / ... via CALVIN_ROOT
#         and CALVIN_HOST_SITE.
#
# numpy is pinned to the *host* evaluation environment's version: freezing a
# different fortran/BLAS build than the one used by the host scipy/pybullet
# mixes two runtimes in one process (SIGSEGV in recompute paths).
#
# Usage: PYTHON_BIN=<host-python-with-caldin-deps> bash scripts/build_pyinstaller.sh
# Output: dist/calvin_benchmark-<version>-linux-x86_64.tar.gz (+ sha256/size)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DIST_DIR=$NODE_ROOT/dist
WORK_DIR=$NODE_ROOT/build/pyinstaller
PYTHON_BIN=${PYTHON_BIN:-python3.12}
ENTRYPOINT=calvin_benchmark
VERSION=${VERSION:-1.0.0}
TMPROOT=${TMPROOT:-${TMPDIR:-/tmp}}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found; set PYTHON_BIN to a python3.12 env with dora/forge_msgs/forge_tool/numpy" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib

for name in ("dora", "forge_msgs", "forge_tool", "pyarrow", "numpy", "yaml", "pydantic", "PIL"):
    importlib.import_module(name)
    print(f"build dependency available: {name}")
PY

BUILD_ENV=$(mktemp -d "$TMPROOT/calvin_benchmark_pyinstaller.XXXXXX")
cleanup() {
    rm -rf "$BUILD_ENV"
}
trap cleanup EXIT

echo "==> creating isolated PyInstaller environment"
"$PYTHON_BIN" -m venv "$BUILD_ENV"

HOST_NUMPY_VERSION=$("$PYTHON_BIN" -c "import numpy; print(numpy.__version__)")
echo "==> host numpy version: $HOST_NUMPY_VERSION (frozen numpy pinned to it)"

"$BUILD_ENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    pyinstaller==6.19.0 \
    "numpy==${HOST_NUMPY_VERSION}" \
    dora-rs==0.4.1 \
    forge-msgs==1.2.0 \
    forge-tool==1.0.0 \
    pyarrow==23.0.1 \
    pydantic==2.13.5 \
    pyyaml==6.0.3 \
    pillow==12.3.0

rm -rf "$DIST_DIR" "$WORK_DIR"
mkdir -p "$DIST_DIR" "$WORK_DIR"

echo "==> building one-file executable (thin freeze)"
CALVIN_NODE_ROOT="$NODE_ROOT" \
    "$BUILD_ENV/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --distpath "$DIST_DIR" \
    --workpath "$WORK_DIR" \
    "$SCRIPT_DIR/calvin_benchmark.spec"

DIST_FILE="$DIST_DIR/$ENTRYPOINT"
if [[ ! -x "$DIST_FILE" ]]; then
    echo "ERROR: executable not found: $DIST_FILE" >&2
    exit 1
fi

echo "==> staging tarball (single entrypoint: $ENTRYPOINT)"
STAGE_DIR=$(mktemp -d "$TMPROOT/calvin_benchmark_stage.XXXXXX")
cp "$DIST_FILE" "$STAGE_DIR/$ENTRYPOINT"
TARBALL="$DIST_DIR/${ENTRYPOINT}-${VERSION}-linux-x86_64.tar.gz"
tar -czf "$TARBALL" -C "$STAGE_DIR" "$ENTRYPOINT"
rm -rf "$STAGE_DIR"

# Single-entrypoint assertion: the archive must contain exactly one root entry
# whose name equals the entrypoint.  Last gate before delivery.
ROOT_ENTRIES=$(tar -tzf "$TARBALL")
if [[ "$ROOT_ENTRIES" != "$ENTRYPOINT" ]]; then
    echo "ERROR: tarball must contain exactly one root entry '$ENTRYPOINT', got: $ROOT_ENTRIES" >&2
    exit 1
fi

echo "==> build successful: $TARBALL"
ls -lh "$TARBALL"
sha256sum "$TARBALL"
stat -c '%n %s bytes' "$TARBALL"
