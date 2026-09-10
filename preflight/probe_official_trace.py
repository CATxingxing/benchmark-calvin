#!/usr/bin/env python3
"""Deterministic trace: official CALVIN env path vs the Forge CALVIN adapter.

The official MCIL baseline consumes only ``rgb_static`` and an 8D proprio
subset, so the Forge↔offical score comparison (判据 A) cannot exercise the
``image/second`` (gripper) channel nor the full native 15D ``robot_obs``.  This
probe drives the *official* env construction (``get_env`` +
``get_env_state_for_initial_condition``, exactly what
``calvin_agent.wrappers.calvin_env_wrapper.CalvinEnvWrapper`` does) and the
Forge ``CalvinAdapter`` through the same reset and the same fixed action
sequence, and compares every frame bit-for-bit.

It is read-only: it never edits CALVIN, the shared core or the adapter.
"""

from __future__ import annotations

import argparse
import hashlib
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

OBS_SPACE = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def _as_uint8_carrier(arr: np.ndarray) -> np.ndarray:
    """The Forge Image carrier is 8-bit; the CALVIN EGL renderer exposes the
    same 0..255 pixels as a wider integer dtype.  Compare the lossless uint8
    representation (values must already be in range)."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 255):
            raise ValueError(f"pixel value out of uint8 range: {arr.dtype} {arr.min()}..{arr.max()}")
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _frame(obs: dict, adapter: CalvinAdapter | None) -> dict:
    if adapter is None:
        rgb = obs["rgb_obs"]
        return {
            "robot_obs": np.asarray(obs["robot_obs"]),
            "scene_obs": np.asarray(obs["scene_obs"]),
            "image/main": _as_uint8_carrier(rgb["rgb_static"]),
            "image/second": _as_uint8_carrier(rgb["rgb_gripper"]),
        }
    return {
        "robot_obs": adapter.to_state(obs),
        "scene_obs": np.asarray(obs["scene_obs"]),
        "image/main": adapter.to_images(obs)["image/main"],
        "image/second": adapter.to_images(obs)["image/second"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.task_ids = [args.task]
    cfg.init_state_ids = [0]

    # ---- Forge side (the adapter under test) ----
    adapter = CalvinAdapter(cfg)
    initial_state, sequence = adapter._sequences[args.task]
    obs_forge = adapter.reset_episode(args.task, 0, 0, episode_seed=0)

    # ---- Official side (upstream construction, same process) ----
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from calvin_env.envs.play_table_env import get_env

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    env = get_env(adapter.validation_path, obs_space=OBS_SPACE, show_gui=False)
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    obs_official = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    fixed_action = np.array([0.05, -0.20, 0.55, 3.10, 0.00, 1.50, -1.0], dtype=np.float32)

    mismatches = []
    records = []

    def compare(step: int) -> None:
        fo = _frame(obs_official, None)
        ff = _frame(obs_forge, adapter)
        row = {"step": step}
        for key in ("robot_obs", "scene_obs", "image/main", "image/second"):
            same = np.array_equal(fo[key], ff[key])
            row[f"{key}_equal"] = bool(same)
            row[f"{key}_sha_official"] = _sha(fo[key])
            row[f"{key}_sha_forge"] = _sha(ff[key])
            if not same:
                mismatches.append({"step": step, "field": key})
        records.append(row)

    compare(0)
    for step in range(1, args.steps + 1):
        # Official env consumes the split container; the adapter's codec builds
        # the identical container from the flat action.
        container = [
            fixed_action[:3].copy(),
            fixed_action[3:6].copy(),
            fixed_action[6:].copy(),
        ]
        obs_official, _, _, _ = env.step(container)
        adapter.step(fixed_action)
        obs_forge = adapter._obs
        compare(step)

    # TaskOracle on the same start/end snapshots for the first subtask.
    oracle = adapter._oracle
    start_official = None
    # Re-derive start snapshot: reset both again to the same initial condition.
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    start_official = env.get_info()
    obs_forge = adapter.reset_episode(args.task, 0, 0, episode_seed=0)
    start_forge = adapter._env.get_info()
    subtask = sequence[0]
    oracle_official = sorted(oracle.get_task_info_for_set(start_official, env.get_info(), {subtask}))
    oracle_forge = sorted(oracle.get_task_info_for_set(start_forge, adapter._env.get_info(), {subtask}))

    env.close()
    adapter.close()

    report = {
        "task": args.task,
        "sequence": list(sequence),
        "steps": args.steps,
        "all_frames_bit_identical": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "first_and_last_frame": [records[0], records[-1]],
        "oracle_subtask": subtask,
        "oracle_official": oracle_official,
        "oracle_forge": oracle_forge,
        "oracle_equal": oracle_official == oracle_forge,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
