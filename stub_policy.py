#!/usr/bin/env python3
"""Deterministic fake policy for CALVIN benchmark lockstep smoke tests.

It consumes the same observation/lifecycle channels as a Forge Policy node and
emits exactly one JointCommand after each complete observation.  It is only a
scheduling smoke tool; it does not attempt to solve CALVIN tasks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from dora import Node
from forge_msgs import JointCommand, JointState


def _action_mode() -> str:
    config_path = os.environ.get("BENCHMARK_CONFIG")
    if not config_path:
        return "relative_euler"
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return str(data.get("action_mode", "relative_euler"))


def _make_action(mode: str, state: JointState | None) -> np.ndarray:
    if mode == "relative_euler":
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0
        return action
    if mode == "absolute_euler":
        if state is None:
            raise RuntimeError("absolute_euler stub requires proprio_state before image/second")
        values = np.asarray(state.position, dtype=np.float32).reshape(-1)
        names = list(state.name)
        required = ["tcp_pos_x", "tcp_pos_y", "tcp_pos_z", "tcp_orn_x", "tcp_orn_y", "tcp_orn_z"]
        index = {name: i for i, name in enumerate(names)}
        missing = [name for name in required if name not in index]
        if missing:
            raise RuntimeError(
                "absolute_euler stub requires Euler proprio labels; missing: " + ", ".join(missing)
            )
        action = np.asarray([values[index[name]] for name in required] + [-1.0], dtype=np.float32)
        if "gripper_action" in index:
            action[-1] = 1.0 if values[index["gripper_action"]] > 0 else -1.0
        return action
    raise RuntimeError(
        "stub_policy supports relative_euler and absolute_euler only; "
        f"got {mode!r}"
    )


def main() -> int:
    node = Node()
    mode = _action_mode()
    latest_state: JointState | None = None
    observations = 0
    actions = 0

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            if event_id == "proprio_state":
                latest_state = JointState.from_arrow(event["value"])
            elif event_id == "image/second":
                # benchmark publishes state -> image/main -> image/second; replying
                # on image/second gives exactly one action per complete observation.
                observations += 1
                action = _make_action(mode, latest_state)
                names = [f"action_{i}" for i in range(action.size)]
                node.send_output(
                    "action",
                    JointCommand.from_np(action, names, "position").to_arrow(),
                )
                actions += 1
        elif event["type"] == "STOP":
            break

    print(f"[calvin_stub] obs={observations} act={actions}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
