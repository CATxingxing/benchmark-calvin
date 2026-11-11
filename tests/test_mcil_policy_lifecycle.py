"""Exercise the real Arrow/Dora boundary with a deterministic fake model."""
import itertools
import numpy as np
import pytest
from forge_msgs import Image, JointCommand, JointState, PolicyCommand
from real_policy import RealPolicyNode


class Runtime:
    def __init__(self):
        self.resets = 0
        self.calls = []
        self.instruction = None
        self.action = np.arange(7, dtype=np.float32)

    def set_instruction(self, text):
        if text != "open drawer":
            raise KeyError(text)
        self.instruction = text

    def reset(self):
        self.resets += 1

    def step(self, *obs):
        self.calls.append(obs)
        return self.action


class Node:
    def __init__(self):
        self.actions = []

    def send_output(self, channel, value):
        assert channel == "action"
        self.actions.append(JointCommand.from_arrow(value))


def command(policy, node, cmd, policy_id="calvin_policy"):
    value = PolicyCommand.from_inputs(policy_id=policy_id, command=cmd, inputs={"instruction": "open drawer"}).to_arrow()
    return policy.process_event(node, {"type": "INPUT", "id": "policy_command", "value": value})


def observation(channel, value=1):
    if channel == "proprio_state":
        batch = JointState.from_np(np.full(15, value, dtype=np.float64), [f"j{i}" for i in range(15)], "position").to_arrow()
    else:
        batch = Image.from_numpy(np.full((8, 8, 3), value, dtype=np.uint8), "rgb8").to_arrow()
    return {"type": "INPUT", "id": channel, "value": batch}


@pytest.mark.parametrize("channels", list(itertools.permutations(("proprio_state", "image/main", "image/second"))))
def test_one_action_per_complete_frame_in_any_channel_order(channels):
    policy, node = RealPolicyNode(Runtime()), Node()
    for cmd in ("start", "set_instruction", "reset_scene"):
        command(policy, node, cmd)
    for index, channel in enumerate(channels):
        policy.process_event(node, observation(channel))
        assert len(node.actions) == int(index == 2)
    policy.process_event(node, {"type": "INPUT", "id": "tick", "value": None})
    assert len(node.actions) == 1
    assert len(policy.runtime.calls) == 1


def test_reset_barrier_preserves_observation_arriving_before_reset_command():
    policy, node = RealPolicyNode(Runtime()), Node()
    command(policy, node, "start")
    command(policy, node, "set_instruction")
    for ch in ("image/second", "proprio_state", "image/main"):
        policy.process_event(node, observation(ch))
    assert not node.actions
    command(policy, node, "reset_scene")
    assert len(node.actions) == 1
    assert policy.runtime.resets == 1


def test_stop_keeps_process_alive_and_allows_second_session_without_stale_frames():
    policy, node = RealPolicyNode(Runtime()), Node()
    for run in range(2):
        for cmd in ("start", "set_instruction", "reset_scene"):
            assert command(policy, node, cmd)
        for ch in ("proprio_state", "image/main", "image/second"):
            policy.process_event(node, observation(ch, run + 1))
        assert len(node.actions) == run + 1
        assert command(policy, node, "stop")
        assert not policy.running and not policy.latest
        policy.process_event(node, {"type": "INPUT", "id": "tick", "value": None})
        assert len(node.actions) == run + 1
    assert [obs[0][0] for obs in policy.runtime.calls] == [1, 2]
    assert not policy.process_event(node, {"type": "STOP"})


def test_wrong_policy_id_cannot_start_or_reset_this_policy():
    policy, node = RealPolicyNode(Runtime()), Node()
    for cmd in ("start", "set_instruction", "reset_scene"):
        command(policy, node, cmd, "someone_else")
    assert not policy.running and policy.runtime.resets == 0


@pytest.mark.parametrize("action", [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)])
def test_invalid_action_is_rejected_before_publish(action):
    policy, node = RealPolicyNode(Runtime()), Node()
    policy.runtime.action = action
    for cmd in ("start", "set_instruction", "reset_scene"):
        command(policy, node, cmd)
    policy.process_event(node, observation("proprio_state"))
    policy.process_event(node, observation("image/main"))
    with pytest.raises(RuntimeError, match="invalid 7D"):
        policy.process_event(node, observation("image/second"))
    assert not node.actions
