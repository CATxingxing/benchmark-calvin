#!/usr/bin/env bash
# Self-contained Forge Node launcher (one Node = one environment).
#
# Layout (next to this file):  runtime/{uv,pyproject.toml,uv.lock,.python-version,node.env,verify.py,
#                                       patch_site.py,wheels/}
#                              app/            Node source
# Every start: `uv sync --frozen` against runtime/uv.lock into a private venv keyed by the
# lock digest, apply the upstream site-packages patches, verify the installed set, then exec
# the Node. Nothing comes from the host Python. `<node> --prepare` only builds the environment.
set -euo pipefail

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
runtime="$here/runtime"
app="$here/app"
# shellcheck source=/dev/null
source "$runtime/node.env"            # NODE_NAME, NODE_ENTRY, NODE_REQUIRE_CUDA, NODE_CORE_IMPORTS

paos_home="${HOME:?HOME must be set}/.PhyAgentOS"
lock_digest="$({ cat "$runtime/uv.lock"; cat "$runtime/patch_site.py" 2>/dev/null || true; } | sha256sum | cut -c1-16)"
env_root="$paos_home/cache/nodes/$NODE_NAME/$lock_digest"
mkdir -p "$env_root"

# Isolate from whatever the caller's shell had active.
unset VIRTUAL_ENV CONDA_PREFIX PYTHONHOME PYTHONPATH PYTHONSTARTUP CALVIN_HOST_SITE CALVIN_PYTHON
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$env_root/venv"
export UV_PYTHON_INSTALL_DIR="$env_root/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_NO_CONFIG=1
# Download cache only (not an environment); keep it on the same disk as the venv.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$paos_home/cache/uv}"
export NODE_CORE_IMPORTS

log() { printf '[%s] %s\n' "$NODE_NAME" "$*" >&2; }

ensure_env() {
  exec 9>"$env_root/.lock"
  flock 9
  if [[ ! -f "$env_root/.verified" ]]; then
    log "preparing private environment $env_root (first start downloads packages)"
  fi
  "$runtime/uv" sync --project "$runtime" --frozen --no-dev --no-install-project \
      --python-preference only-managed --quiet >&2
  if [[ -f "$runtime/patch_site.py" ]]; then
    "$env_root/venv/bin/python" -I "$runtime/patch_site.py" >&2
  fi
  "$env_root/venv/bin/python" -I "$runtime/verify.py" "$runtime/uv.lock" "$NODE_REQUIRE_CUDA" >&2
  touch "$env_root/.verified"
  flock -u 9
}

ensure_env
[[ "${1:-}" == "--prepare" ]] && { log "environment ready"; exit 0; }

export PYTHONPATH="$app"
exec "$env_root/venv/bin/python" -s "$app/$NODE_ENTRY" "$@"
