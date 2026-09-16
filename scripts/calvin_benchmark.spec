# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-file specification for the CALVIN benchmark node.

Thin-freeze recipe (ported from the official ``libero_benchmark`` recipe and the
RoboTwin thin-node recipe):

* Frozen: the node code plus the lightweight protocol dependencies
  (``dora`` / ``forge_msgs`` / ``forge_tool`` / ``numpy`` / ``pyarrow`` /
  ``pydantic`` / ``yaml`` / ``PIL``).
* Host-provided (NOT frozen, injected into ``sys.path`` at runtime by
  ``adapters/calvin_adapter.py`` via ``CALVIN_ROOT`` + ``CALVIN_HOST_SITE``):
  ``calvin_env`` / ``calvin_models`` / ``calvin_agent`` / ``pybullet`` /
  ``torch`` / ``scipy`` / ``hydra`` / ``omegaconf`` / ... .

Freezing the CALVIN stack into the bundle is deliberately avoided: it would (a)
drag ``calvin_agent`` -> ``pytorch_lightning`` into the import chain and (b)
place both ``calvin_env/conf`` and ``calvin_models/conf`` under the same
onefile root, where the ``parents[1]/conf`` lookups would collide.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


_node_root = Path(os.environ["CALVIN_NODE_ROOT"]).resolve()
_entry = _node_root / "scripts" / "entrypoint.py"
_hooks = _node_root / "scripts" / "hooks"


def _without_tests(name: str) -> bool:
    return (
        ".tests" not in name
        and not name.endswith(".test")
        and not name.endswith("_test")
        and not name.endswith("conftest")
    )


hiddenimports = [
    "_core.adapter",
    "_core.config",
    "_core.node",
    "_core.result",
    "_core.runner",
    "_core.session",
    "_core.video",
    "adapters.calvin_adapter",
    "bench_tool",
    "bench_tool.descriptor",
    "bench_tool.endpoints",
    "bench_tool.invocations",
    "bench_tool.lease",
    "bench_tool.node",
    "bench_tool.result_shape",
    "numpy.core._multiarray_tests",
]

# numpy must be collected in full: once the adapter injects the host
# site-packages, host code (calvin_env / pybullet / ...) imports numpy
# submodules that must resolve to the frozen copy, never to a second host copy
# (avoids "cannot load module more than once per process" / SIGSEGV).
for package in ("dora", "forge_msgs", "forge_tool", "yaml", "pydantic", "numpy"):
    hiddenimports.extend(
        module for module in collect_submodules(package) if _without_tests(module)
    )

# Partially frozen stdlib packages must be completed: once a package is frozen
# it is anchored, and host code importing a missing submodule will not fall back
# to the host stdlib.  (Same list proven by the RoboTwin thin node.)
for package in (
    "asyncio",
    "concurrent",
    "ctypes",
    "email",
    "html",
    "http",
    "importlib",
    "json",
    "logging",
    "multiprocessing",
    "pydoc_data",
    "tomllib",
    "unittest",
    "urllib",
    "xml",
    "xmlrpc",
    "zipfile",
    "zoneinfo",
):
    hiddenimports.extend(collect_submodules(package))

# forge_msgs pulls PIL into the frozen chain; complete it (GUI bindings that
# drag tkinter/Qt are excluded).
hiddenimports.extend(
    module
    for module in collect_submodules("PIL")
    if _without_tests(module)
    and not module.startswith(("PIL.ImageTk", "PIL.ImageQt"))
)

datas = []
for package in ("dora", "forge_msgs", "pydantic", "yaml"):
    datas.extend(collect_data_files(package))

binaries = []
for package in ("dora", "forge_msgs", "forge_tool", "pyarrow"):
    binaries.extend(collect_dynamic_libs(package))

# NumPy links against BLAS/LAPACK/Fortran support libraries that live next to
# the interpreter.  Include them by discovered prefix, never by checkout path.
_runtime_lib_dir = Path(getattr(sys, "base_prefix", sys.prefix)) / "lib"
for _pattern in (
    "libblas.so*",
    "liblapack.so*",
    "libopenblas*.so*",
    "libgfortran.so*",
    "libquadmath.so*",
    "libgcc_s.so*",
):
    for _library in _runtime_lib_dir.glob(_pattern):
        binaries.append((str(_library), "."))

a = Analysis(
    [str(_entry)],
    pathex=[str(_node_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(_hooks)],
    runtime_hooks=[],
    hooksconfig={},
    excludes=[
        # Host-provided simulation / ML stack (thin node boundary).
        "calvin_env",
        "calvin_models",
        "calvin_agent",
        "pybullet",
        "scipy",
        "numba",
        "hydra",
        "omegaconf",
        "torch",
        "torchvision",
        "pytorch_lightning",
        "torchmetrics",
        "matplotlib",
        "OpenGL",
        "trimesh",
        "urdfpy",
        "pyrender",
        "pyglet",
        "gym",
        "pandas",
        "cv2",
        "tacto",
        "eglRenderer",
        "sentence_transformers",
        "wandb",
        "pynvml",
        "psutil",
        "git",
        "gitdb",
        "smmap",
        "tqdm",
        "networkx",
        "sklearn",
        "rospkg",
        "pyhash",
        "curobo",
        # Tests.
        "pytest",
        "tests",
        "pyarrow.tests",
        "pyarrow.conftest",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="calvin_benchmark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
