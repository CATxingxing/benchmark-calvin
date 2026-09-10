#!/usr/bin/env python3
"""Read-only real-environment probes for the CALVIN benchmark adapter.

These probes exercise the *actual* adapter path (the same ``_ensure_env`` /
``reset_episode`` the node uses) and only observe.  They never edit CALVIN,
the shared ``_core`` or the environment.  Evidence is written under
``evidence/preflight/`` (gitignored).

Checks (manual §1.1 premise C, §3.2/§3.3 observation contract, §7 trap 2):
  1. observation/state contract shapes and dtypes;
  2. TaskOracle call path returns a set (empty before any progress);
  3. same (task, init_state) reset twice -> robot_obs/scene_obs and both RGB
     cameras bit-identical (no hidden per-reset layout resampling);
  4. dump one native frame per camera for a manual orientation check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adapters.calvin_adapter import CalvinAdapter  # noqa: E402
from config import load_config  # noqa: E402


def _image_report(img: np.ndarray) -> dict:
    arr = np.asarray(img)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }


def main() -> int:
    config_path = os.environ.get("BENCHMARK_CONFIG", str(_ROOT / "config.yaml"))
    cfg = load_config(config_path)
    cfg.protocol = "lh_mtlc"

    adapter = CalvinAdapter(cfg)
    out_dir = _ROOT / "evidence" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    task_id = int(cfg.task_ids[0])
    obs_a = adapter.reset_episode(task_id, 0, 0, episode_seed=12345)
    state_a = adapter.to_state(obs_a)
    imgs_a = adapter.to_images(obs_a)
    info_a = adapter._env.get_info()

    # Two-state determinism: reset the exact same episode identity again in the
    # same process and require bit-identical state and pixels.
    obs_b = adapter.reset_episode(task_id, 0, 0, episode_seed=12345)
    state_b = adapter.to_state(obs_b)
    imgs_b = adapter.to_images(obs_b)

    scene_a = np.asarray(obs_a["scene_obs"])
    scene_b = np.asarray(obs_b["scene_obs"])

    report: dict = {
        "config": str(config_path),
        "task_id": task_id,
        "sequence": list(adapter._sequences[task_id][1]),
        "initial_condition": adapter._sequences[task_id][0],
        "state": {
            "dim": int(state_a.shape[0]),
            "names": adapter.state_joint_names(),
            "bit_identical_on_rereset": bool(np.array_equal(state_a, state_b)),
        },
        "scene_obs": {
            "dim": int(scene_a.shape[0]),
            "bit_identical_on_rereset": bool(np.array_equal(scene_a, scene_b)),
        },
        "images": {
            name: _image_report(img) for name, img in imgs_a.items()
        },
        "images_bit_identical_on_rereset": {
            name: bool(np.array_equal(imgs_a[name], imgs_b[name])) for name in imgs_a
        },
        "robot_obs_raw_dim": int(np.asarray(obs_a["robot_obs"]).shape[0]),
        "info_keys": sorted(info_a.keys()),
    }

    # Oracle call path: compare start snapshot to itself -> no task can be newly
    # satisfied, so the returned set must be empty.  This proves the call works
    # without needing a policy to actually solve something.
    task_oracle = adapter._oracle
    first_task = adapter._sequences[task_id][1][0]
    same = task_oracle.get_task_info_for_set(info_a, info_a, {first_task})
    report["oracle"] = {
        "subtask": first_task,
        "same_snapshot_result_type": type(same).__name__,
        "same_snapshot_result": sorted(same),
        "call_path_ok": isinstance(same, set),
    }

    # Persist one native frame per camera for the manual orientation check.
    try:
        import cv2

        frames = {}
        for name, img in imgs_a.items():
            path = out_dir / f"frame_{name.replace('/', '_')}.png"
            cv2.imwrite(str(path), np.asarray(img)[:, :, ::-1])
            frames[name] = str(path)
        report["frame_files"] = frames
    except Exception as exc:  # noqa: BLE001
        report["frame_files_error"] = f"{type(exc).__name__}: {exc}"

    adapter.close()

    out_path = out_dir / "probe_env_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
