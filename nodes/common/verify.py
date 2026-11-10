"""Verify a Node's private environment against its uv.lock before the Node starts.

Usage: NODE_CORE_IMPORTS="a b c" python -I verify.py <uv.lock> <require_cuda 0|1>
Fails (exit 1) on: wrong interpreter, any locked package missing or at another
version, a core import failing, or CUDA unavailable when the Node needs it.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import re
import sys
import tomllib


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    lock_path, require_cuda = sys.argv[1], sys.argv[2] == "1"
    core_imports = os.environ.get("NODE_CORE_IMPORTS", "").split()
    errors: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append(f"interpreter is {sys.version.split()[0]}, Node requires 3.12")

    with open(lock_path, "rb") as stream:
        lock = tomllib.load(stream)
    installed = {_norm(d.metadata["Name"]): d.version for d in md.distributions()}
    checked = 0
    for pkg in lock.get("package", []):
        source = pkg.get("source", {})
        if "virtual" in source or "editable" in source:
            continue  # the Node project itself is never installed
        name, want = _norm(pkg["name"]), pkg["version"]
        have = installed.get(name)
        checked += 1
        if have is None:
            errors.append(f"{name}=={want} is not installed")
        elif have.split("+")[0] != want.split("+")[0]:
            errors.append(f"{name} is {have}, lock requires {want}")

    for module in core_imports:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report every broken import
            errors.append(f"import {module} failed: {exc}")

    if require_cuda and not errors:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable (check CUDA_VISIBLE_DEVICES and the driver)")
        else:
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            if arch not in torch.cuda.get_arch_list():
                errors.append(f"torch {torch.__version__} has no kernels for {arch}")

    if errors:
        print("Node environment verification failed:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    print(f"environment verified: {checked} locked packages, {len(core_imports)} core imports, "
          f"python {sys.version.split()[0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
