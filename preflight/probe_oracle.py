#!/usr/bin/env python3
"""Read-only probe: confirm the CALVIN TaskOracle call path (manual §6 step 2).

The official evaluator decides success ONLY via
``Tasks.get_task_info_for_set(start_info, current_info, {subtask})``.  This probe
verifies that the path used by the adapter returns a ``set`` after a reset, and
that a no-op comparison (start vs start) is empty.  It never edits CALVIN, the
core or the adapter.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adapters.calvin_adapter import CalvinAdapter  # noqa: E402
from config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config(os.environ.get("BENCHMARK_CONFIG", str(_ROOT / "config.yaml")))
    cfg.task_ids = [int(os.environ.get("TASK_ID", "0"))]
    cfg.init_state_ids = [0]
    adapter = CalvinAdapter(cfg)
    task = int(cfg.task_ids[0])

    adapter.reset_episode(task, 0, 0, episode_seed=0)
    env = adapter._env
    start_info = env.get_info()
    subtask = adapter._sequences[task][1][0]
    same = adapter._oracle.get_task_info_for_set(start_info, start_info, {subtask})

    report = {
        "success_source": CalvinAdapter.SUCCESS_SOURCE,
        "call": "Tasks.get_task_info_for_set(start_info, current_info, {subtask})",
        "subtask": subtask,
        "returns_set": isinstance(same, set),
        "start_vs_start_empty": len(same) == 0,
        "info_keys": sorted(start_info.keys()),
    }
    adapter.close()
    out = _ROOT / "evidence" / "preflight" / "probe_oracle_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
