#!/usr/bin/env python3
"""CALVIN benchmark node (Dora).

The generic Forge benchmark core is vendored from benchmark-libero with the
multi-suite extension listed in README.md; CALVIN-specific semantics live in
adapters/calvin_adapter.py and config.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _core.node import run_node  # noqa: E402
from adapters.calvin_adapter import CalvinAdapter  # noqa: E402
from config import load_config  # noqa: E402


def _log(msg: str) -> None:
    print(f"[calvin_bench] {msg}", flush=True)


def main() -> int:
    config_path = os.environ.get("BENCHMARK_CONFIG", str(_HERE / "config.yaml"))
    cfg = load_config(config_path)
    adapter = CalvinAdapter(cfg)
    if getattr(cfg, "serve", False):
        from bench_tool.node import run_serve_tool  # noqa: E402
        return run_serve_tool(cfg, adapter, log=_log)
    return run_node(cfg, adapter, log=_log)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
