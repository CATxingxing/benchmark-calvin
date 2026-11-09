#!/usr/bin/env python3
"""MCIL policy node for the CALVIN evaluation Skill (source and thin binary)."""
from __future__ import annotations
import os
from typing import Any

import numpy as np
from dora import Node
from forge_msgs import Image as ForgeImage
from forge_msgs import JointCommand, JointState, PolicyCommand

ACTION_NAMES = [f"action_{i}" for i in range(7)]


def _env_path(name: str) -> str:
    """Dataflow env values are passed verbatim, so expand ``~`` here."""
    return os.path.expanduser(os.environ[name])


OBS_CHANNELS = {"proprio_state": "robot_obs", "image/main": "rgb_static", "image/second": "rgb_gripper"}


class RealPolicyNode:
    def __init__(self, runtime: Any = None, policy_id: str | None = None):
        self.policy_id = policy_id or os.environ.get("POLICY_ID", "calvin_policy")
        if runtime is None:
            from mcil_runtime import MCILRuntime
            runtime = MCILRuntime(_env_path("MCIL_TRAIN_FOLDER"), _env_path("MCIL_CHECKPOINT"),
                                  _env_path("CALVIN_DATASET"), os.environ.get("POLICY_DEVICE", "cuda:0"))
        self.runtime = runtime
        self.running = False
        self.instruction: str | None = None
        self.latest: dict[str, np.ndarray] = {}
        self.action_count = 0
        self._reset_pending = True

    def _handle_command(self, value: Any) -> None:
        cmd = PolicyCommand.from_arrow(value)
        if cmd.policy_id != self.policy_id:
            return
        if cmd.command == "start":
            self.running = True
        elif cmd.command == "set_instruction":
            instruction = str((cmd.inputs() or {}).get("instruction", ""))
            self.runtime.set_instruction(instruction)
            self.instruction = instruction
            self._reset_pending = True
        elif cmd.command == "reset_scene":
            self.runtime.reset()
            # Preserve a new complete observation that beat reset_scene across
            # Dora channels. The previous frame was consumed/cleared on send.
            self._reset_pending = False
        elif cmd.command == "stop":
            # A benchmark Session ending is not the end of the policy process.
            self.running = False
            self.instruction = None
            self._reset_pending = True
            self.latest.clear()
            self.runtime.reset()

    def _maybe_act(self, node: Any) -> None:
        if not self.running or self.instruction is None or self._reset_pending:
            return
        if not set(OBS_CHANNELS.values()) <= self.latest.keys():
            return
        action = np.asarray(self.runtime.step(self.latest["robot_obs"], self.latest["rgb_static"], self.latest["rgb_gripper"]), dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.all(np.isfinite(action)):
            raise RuntimeError(f"MCIL returned an invalid 7D action: {action.shape}")
        node.send_output("action", JointCommand.from_np(action, ACTION_NAMES, "position").to_arrow())
        self.latest.clear()
        self.action_count += 1

    def process_event(self, node: Any, event: Any) -> bool:
        if event["type"] == "STOP":
            return False
        if event["type"] == "ERROR":
            raise RuntimeError(f"Dora policy error: {event.get('error', event)}")
        if event["type"] != "INPUT":
            return True
        channel = event["id"]
        if channel == "policy_command":
            self._handle_command(event["value"])
        elif channel in OBS_CHANNELS:
            if channel == "proprio_state":
                data = np.asarray(JointState.from_arrow(event["value"]).position, dtype=np.float64)
            else:
                data = np.ascontiguousarray(ForgeImage.from_arrow(event["value"]).to_numpy())
            self.latest[OBS_CHANNELS[channel]] = data
        self._maybe_act(node)
        return True

    def run(self, node: Any) -> int:
        from policy_tool import PolicyTool
        tool = PolicyTool(node, lambda: {"policy_id": self.policy_id, "backend": "mcil",
                                       "ready": True, "running": self.running, "action_dim": 7}) if os.environ.get("POLICY_TOOL_ENABLED") == "1" else None
        try:
            for event in node:
                if tool:
                    tool.tick()
                    if event["type"] == "INPUT" and event["id"] == "tool_in":
                        tool.process(event["value"])
                        continue
                if not self.process_event(node, event):
                    break
        finally:
            if tool:
                tool.close()
        return 0


def main() -> int:
    import torch
    from pytorch_lightning import seed_everything
    seed_everything(int(os.environ.get("POLICY_SEED", "0")), workers=True)
    torch.backends.cudnn.deterministic = True
    node = Node()
    return RealPolicyNode().run(node)


if __name__ == "__main__":
    raise SystemExit(main())
