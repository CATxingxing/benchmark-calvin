from __future__ import annotations

import os
import importlib
from types import SimpleNamespace

import numpy as np


from adapters.calvin_adapter import CalvinActionCodec, CalvinAdapter, LhMtlcProtocol, MtlcProtocol


def _core_runner():
    import importlib

    return importlib.import_module("_core.runner")


def _core_config():
    import importlib

    return importlib.import_module("_core.config")


def test_action_codec_is_explicit_and_native():
    with np.testing.assert_raises(ValueError):
        CalvinActionCodec("unknown")
    codec = CalvinActionCodec("absolute_euler")
    encoded = codec.encode([1, 2, 3, 4, 5, 6, 0])
    assert isinstance(encoded, list)
    assert [x.tolist() for x in encoded] == [[1, 2, 3], [4, 5, 6], [-1.0]]
    assert CalvinActionCodec("relative_euler").encode(np.zeros(7)).shape == (7,)


def test_lh_mtlc_360_boundary_and_1800_total():
    class Oracle:
        def __init__(self):
            self.calls = 0

        def get_task_info_for_set(self, *_args):
            self.calls += 1
            return {"ok"} if self.calls % 360 == 0 else set()

    protocol = LhMtlcProtocol(tuple("abcde"), lambda task: task)
    protocol.begin({})
    oracle = Oracle()
    for count in range(1, 1801):
        result = protocol.observe({}, oracle)
        if result["done"]:
            break
    assert count == 1800
    assert result["success"] is True
    assert result["chain_length"] == 5
    assert oracle.calls == 1800


def test_lh_mtlc_subtask_transition_and_metrics_denominator():
    class Oracle:
        def get_task_info_for_set(self, *_args):
            return {"solved"}

    protocol = LhMtlcProtocol(("a", "b", "c", "d", "e"), lambda task: f"instruction:{task}")
    protocol.begin({"state": 0})
    first = protocol.observe({"state": 1}, Oracle())
    assert first == {
        "success": False,
        "done": False,
        "next_instruction": "instruction:b",
        "chain_length": 1,
        "subtasks_completed": 1,
    }

    # Adapter aggregation uses only completed tasks plus the first failed task
    # as denominators; later unattempted tasks never enter per_task.
    cfg = SimpleNamespace(protocol="lh_mtlc")
    adapter = object.__new__(CalvinAdapter)
    adapter.cfg = cfg
    primary, metrics = adapter.aggregate([
        {"chain_length": 2, "subtasks": ["a", "b", "c", "d", "e"], "failed_subtask": "c"},
        {"chain_length": 5, "subtasks": ["a", "b", "c", "d", "e"]},
    ])
    assert primary["value"] == 3.5
    assert metrics["per_task"]["a"] == {"success": 2, "total": 2}
    assert metrics["per_task"]["c"] == {"success": 1, "total": 2}
    assert "d" in metrics["per_task"] and "e" in metrics["per_task"]


def test_mtlc_uses_checkpoint_episode_ids_and_allows_nonzero_init_state():
    assert MtlcProtocol({"open_drawer": [1, 2]}).episode_ids("open_drawer") == (1, 2)
    with np.testing.assert_raises(ValueError):
        MtlcProtocol({"open_drawer": 1}).episode_ids("open_drawer")


def test_shared_core_lockstep_runs_1800_and_never_1801(tmp_path):
    from forge_msgs import JointCommand

    runner_mod = _core_runner()
    config_mod = _core_config()
    step_outcome = importlib.import_module("_core.adapter").StepOutcome

    class Node:
        def __init__(self):
            self.outputs = []

        def send_output(self, name, value):
            self.outputs.append(name)

    class Adapter:
        def __init__(self):
            self.calls = 0

        def state_joint_names(self):
            return ["joint"]

        def task_language(self, _task):
            return "chain"

        def episode_instruction(self, _task, _init, _run):
            return "first"

        def num_tasks(self):
            return 1

        def reset_episode(self, **_kwargs):
            return {"robot_obs": np.zeros(1), "rgb_obs": {"rgb_static": np.zeros((1, 1, 3), dtype=np.uint8), "rgb_gripper": np.zeros((1, 1, 3), dtype=np.uint8)}}

        def to_state(self, _obs):
            return np.zeros(1)

        def to_images(self, _obs):
            return {"image/main": np.zeros((1, 1, 3), dtype=np.uint8), "image/second": np.zeros((1, 1, 3), dtype=np.uint8)}

        def step(self, _action):
            self.calls += 1
            n = self.calls
            success = n == 1800
            transition = n in {360, 720, 1080, 1440}
            return step_outcome(
                obs={"robot_obs": np.zeros(1), "rgb_obs": {"rgb_static": np.zeros((1, 1, 3), dtype=np.uint8), "rgb_gripper": np.zeros((1, 1, 3), dtype=np.uint8)}},
                reward=0.0,
                done=success,
                success=success,
                metrics={"chain_length": n // 360, "subtasks": ["a", "b", "c", "d", "e"]},
                next_instruction="next" if transition else None,
            )

        def close(self):
            pass

        def aggregate(self, metrics):
            return ({"name": "avg_sequence_length", "value": 5.0, "max": 5.0}, {})

    cfg = config_mod.BaseBenchmarkConfig(
        benchmark="calvin", suite="lh_mtlc", task_ids=[0], init_state_ids=[0], max_steps=1800,
        policy_id="fake", result_dir=str(tmp_path), record_dir=None,
    )
    adapter = Adapter()
    node = Node()
    runner = runner_mod.BenchmarkRunner(node, cfg, adapter, log=lambda _msg: None)
    runner.advance()
    command = JointCommand(name=["joint"], mode="position", position=[0.0])
    for _ in range(1800):
        runner.on_action(command)
    runner.on_action(command)  # must be ignored after finish
    assert adapter.calls == 1800
    assert runner.finished is True
    assert len(runner.result.episodes) == 1
    assert runner.result.episodes[0].success is True


def test_adapter_observation_contract_without_policy(tmp_path):
    validation = tmp_path / "validation"
    hydra_dir = validation / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "merged_config.yaml").write_text(
        "robot:\n  euler_obs: true\n  arm_joint_ids: [0, 1, 2, 3, 4, 5, 6]\n",
        encoding="utf-8",
    )

    cfg = SimpleNamespace(
        dataset_path=str(tmp_path),
        action_mode="absolute_euler",
        protocol="lh_mtlc",
        sequence_count=1,
    )
    adapter = CalvinAdapter(
        cfg,
        env=object(),
        sequences=[({}, ["open_drawer"] * 5)],
        oracle=object(),
        annotations={"open_drawer": ["open drawer"]},
    )
    obs = {
        "robot_obs": np.zeros(15),
        "scene_obs": np.zeros(24),
        "rgb_obs": {
            "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        },
    }
    assert adapter.to_state(obs).shape == (15,)
    images = adapter.to_images(obs)
    assert images["image/main"].shape == (200, 200, 3)
    assert images["image/second"].shape == (84, 84, 3)
