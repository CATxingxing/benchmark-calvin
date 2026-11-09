#!/usr/bin/env python3
"""Build the self-contained CALVIN Node archives (calvin_benchmark, calvin_policy).

Each Node is a ``directory_tar_gz`` with its OWN environment definition:

    <node>/<node>      launcher (nodes/common/launcher.sh)
    <node>/runtime/    uv, pyproject.toml, uv.lock, .python-version, verify.py, node.env
    <node>/app/        Node source

On start the launcher runs ``uv sync --frozen`` against ``runtime/uv.lock`` into a
private venv, verifies it, then execs the Node. No host Python, no shared env.

The CALVIN upstream checkout (calvin_env / calvin_models / tacto) is NOT vendored;
it comes from ``CALVIN_ROOT``.

    python3 scripts/build_node_bundles.py --out dist/nodes
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
COMMON_IMPORTS = "numpy torch dora forge_msgs forge_tool pyarrow yaml"
NODES = {
    "calvin_benchmark": {
        "entry": "main.py",
        "imports": COMMON_IMPORTS + " pybullet hydra omegaconf pytorch_lightning cv2 scipy quaternion",
        "files": ["main.py", "config.py", "result.py"],
        "packages": ["_core", "adapters", "bench_tool"],
    },
    "calvin_policy": {
        "entry": "real_policy.py",
        "imports": COMMON_IMPORTS + " hydra omegaconf pytorch_lightning torchvision cv2 scipy",
        "files": ["real_policy.py", "mcil_runtime.py", "policy_tool.py"],
        "packages": ["bench_tool"],
    },
}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".pytest_cache")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_node(node: str, uv: Path, root: Path) -> Path:
    spec = NODES[node]
    top = root / node
    runtime, app = top / "runtime", top / "app"
    node_dir = REPO / "nodes" / node
    runtime.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        shutil.copy2(node_dir / name, runtime / name)
    shutil.copy2(REPO / "nodes" / "common" / "verify.py", runtime / "verify.py")
    shutil.copy2(uv, runtime / "uv")
    (runtime / "node.env").write_text(
        f'NODE_NAME={node}\nNODE_ENTRY={spec["entry"]}\nNODE_REQUIRE_CUDA=1\n'
        f'NODE_CORE_IMPORTS="{spec["imports"]}"\n')
    app.mkdir()
    for name in spec["files"]:
        shutil.copy2(REPO / name, app / name)
    for name in spec["packages"]:
        shutil.copytree(REPO / name, app / name, ignore=IGNORE)
    launcher = top / node
    shutil.copy2(REPO / "nodes" / "common" / "launcher.sh", launcher)
    for path in (launcher, runtime / "uv"):
        path.chmod(0o755)
    return top


def pack(top: Path, archive: Path) -> str:
    def norm(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if (info.isdir() or info.mode & 0o111) else 0o644
        return info

    archive.parent.mkdir(parents=True, exist_ok=True)
    with open(archive, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted([top, *top.rglob("*")]):
            tar.add(path, arcname=path.relative_to(top.parent).as_posix(), recursive=False, filter=norm)
    return sha256(archive)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", type=Path, default=Path(shutil.which("uv") or "uv"))
    parser.add_argument("--out", type=Path, default=REPO / "dist" / "nodes")
    parser.add_argument("--node", choices=sorted(NODES), action="append", help="default: all")
    parser.add_argument("--stage-only", type=Path, help="write the unpacked Nodes here instead of packing")
    args = parser.parse_args()
    nodes = args.node or sorted(NODES)
    if args.stage_only:
        for node in nodes:
            print(stage_node(node, args.uv.resolve(), args.stage_only))
        return 0
    manifest = {}
    for node in nodes:
        with tempfile.TemporaryDirectory(prefix="calvin-nodes-") as tmp:
            top = stage_node(node, args.uv.resolve(), Path(tmp))
            artifact_id = f"{node}-{VERSION}-linux-x86_64"
            archive = args.out / f"{artifact_id}.tar.gz"
            digest = pack(top, archive)
        manifest[node] = {"artifact_id": artifact_id, "version": VERSION, "artifact_type": "directory_tar_gz",
                          "entrypoint": node, "sha256": digest, "file": archive.name,
                          "size": archive.stat().st_size}
        print(f"{archive}  {digest}  {archive.stat().st_size / 1e6:.1f} MB")
    (args.out / "nodes-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
