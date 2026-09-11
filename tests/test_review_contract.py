from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

from adapters.calvin_adapter import CalvinActionCodec, CalvinAdapter, LhMtlcProtocol, MtlcProtocol, _episode_lookup_from_bounds
from config import load_config


def _dataset_root(tmp_path: Path, *, euler: bool = True, arm_ids=(0, 1, 2, 3, 4, 5, 6)) -> Path:
    hydra = tmp_path / "validation" / ".hydra"
    hydra.mkdir(parents=True)
    (hydra / "merged_config.yaml").write_text(
        yaml.safe_dump({"robot": {"euler_obs": euler, "arm_joint_ids": list(arm_ids)}}),
        encoding="utf-8",
    )
    return tmp_path


def _lh_cfg(root: Path) -> SimpleNamespace:
    return SimpleNamespace(dataset_path=str(root), action_mode="absolute_euler", protocol="lh_mtlc", sequence_count=1)


class _Env:
    def __init__(self, dim: int = 15, *, env_done: bool = False):
        self.dim = dim
        self.env_done = env_done

    def _obs(self):
        return {
            "robot_obs": np.zeros(self.dim),
            "scene_obs": np.zeros(24),
            "rgb_obs": {
                "rgb_static": np.zeros((4, 5, 3), dtype=np.uint8),
                "rgb_gripper": np.zeros((2, 3, 3), dtype=np.uint8),
            },
        }

    def reset(self, **_kwargs):
        return self._obs()

    def get_info(self):
        return {"robot_info": {}, "scene_info": {}}

    def step(self, _action):
        return self._obs(), 0.0, self.env_done, {"ignored": True}


class _NeverOracle:
    def get_task_info_for_set(self, *_args):
        return set()


def test_config_forbids_official_evaluator_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    base = {
        "dataset_path": str(tmp_path),
        "protocol": "lh_mtlc",
        "action_mode": "absolute_euler",
        "sequence_count": 1000,
    }
    for key, value in (
        ("annotation_path", "fake.yaml"),
        ("task_config_path", "fake.yaml"),
        ("task_to_id_dict", {"a": [1]}),
        ("euler_obs", False),
    ):
        path.write_text(yaml.safe_dump({**base, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot be overridden"):
            load_config(path)


def test_config_does_not_truncate_sequence_count_float(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_path": str(tmp_path),
                "protocol": "lh_mtlc",
                "action_mode": "absolute_euler",
                "sequence_count": 1000.5,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be an integer"):
        load_config(path)




def test_config_freezes_official_run_shape_and_language(tmp_path):
    base = {
        "dataset_path": str(tmp_path),
        "protocol": "lh_mtlc",
        "suite": "lh_mtlc",
        "action_mode": "absolute_euler",
        "sequence_count": 1000,
        "policy_id": "calvin_policy",
    }
    path = tmp_path / "config.yaml"

    path.write_text(yaml.safe_dump({**base, "num_runs": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="num_runs=1"):
        load_config(path)

    path.write_text(yaml.safe_dump({**base, "send_instruction": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="send_instruction=true"):
        load_config(path)

    path.write_text(yaml.safe_dump({**base, "suite": "mtlc"}), encoding="utf-8")
    with pytest.raises(ValueError, match="suite must match protocol"):
        load_config(path)

    # The reviewed contract requires an explicit suite; it must not default
    # silently to the protocol, otherwise a typo in the protocol would go
    # unnoticed and the result file's suite label could diverge.
    path.write_text(
        yaml.safe_dump({key: value for key, value in base.items() if key != "suite"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="suite is required"):
        load_config(path)

    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.suite == "lh_mtlc"
    assert cfg.num_runs == 1
    assert cfg.send_instruction is True

def test_state_names_follow_dataset_merged_config(tmp_path):
    root = _dataset_root(tmp_path, euler=False, arm_ids=(2, 4))
    adapter = CalvinAdapter(
        _lh_cfg(root),
        env=_Env(dim=11),
        sequences=[({}, ["a"] * 5)],
        oracle=_NeverOracle(),
        annotations={"a": ["do a"]},
    )
    assert adapter.state_joint_names() == [
        "tcp_pos_x", "tcp_pos_y", "tcp_pos_z",
        "tcp_orn_x", "tcp_orn_y", "tcp_orn_z", "tcp_orn_w",
        "gripper_opening_width", "robot_joint_2", "robot_joint_4", "gripper_action",
    ]


def test_lh_adapter_ignores_environment_done():

    adapter = object.__new__(CalvinAdapter)
    adapter.cfg = SimpleNamespace(protocol="lh_mtlc")
    adapter.codec = CalvinActionCodec("absolute_euler")
    adapter._env = _Env(env_done=True)
    adapter._obs = adapter._env.reset()
    adapter._task_id = 0
    adapter._oracle = _NeverOracle()
    adapter._protocol = LhMtlcProtocol(("a", "b", "c", "d", "e"), lambda task: task)
    adapter._protocol.begin(adapter._env.get_info())
    adapter._episode_metrics = {"sequence_length": 5, "chain_length": 0, "subtasks": ["a", "b", "c", "d", "e"]}
    outcome = adapter.step(np.zeros(7))
    assert outcome.success is False
    assert outcome.done is False


def test_mtlc_adapter_ignores_environment_done_before_240():
    adapter = object.__new__(CalvinAdapter)
    adapter.cfg = SimpleNamespace(protocol="mtlc")
    adapter.codec = CalvinActionCodec("absolute_euler")
    adapter._env = _Env(env_done=True)
    adapter._obs = adapter._env.reset()
    adapter._task_id = 0
    adapter._mtlc_trials = [("a", 0)]
    adapter._mtlc_protocol = MtlcProtocol({"a": [0]})
    adapter._mtlc_start_info = {"robot_info": {}, "scene_info": {}}
    adapter._oracle = _NeverOracle()
    adapter._episode_metrics = {"steps": 0, "chain_length": 0, "subtasks": ["a"]}
    outcome = adapter.step(np.zeros(7))
    assert outcome.success is False
    assert outcome.done is False
    assert outcome.metrics["steps"] == 1


def test_mtlc_dataset_index_mapping_matches_disk_dataset_algorithm():
    bounds = np.array([[0, 20], [100, 120]])
    lookup = _episode_lookup_from_bounds(bounds, 16)
    # First episode contributes raw starts 0..4. Dataset index 5 therefore
    # maps to raw frame 100, proving dataset id != filename suffix in general.
    assert lookup.tolist() == [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]
    assert int(lookup[5]) == 100


def test_mtlc_checkpoint_indices_are_strict_integers():
    assert MtlcProtocol({"a": [np.int64(3)]}).episode_ids("a") == (3,)
    with pytest.raises(ValueError, match="integer-like"):
        MtlcProtocol({"a": [3.7]}).episode_ids("a")
    with pytest.raises(ValueError):
        MtlcProtocol({"a": [-1]}).episode_ids("a")


def test_mtlc_trials_flatten_variable_task_counts(monkeypatch, tmp_path):
    root = _dataset_root(tmp_path)
    monkeypatch.setattr(CalvinAdapter, "_load_task_to_id_dict", lambda self: {"a": [10, 11], "b": [20]})
    monkeypatch.setattr(CalvinAdapter, "_prepare_mtlc_dataset_mapping", lambda self: None)
    cfg = SimpleNamespace(dataset_path=str(root), action_mode="absolute_euler", protocol="mtlc", checkpoint_path="unused")
    adapter = CalvinAdapter(
        cfg,
        env=_Env(),
        oracle=_NeverOracle(),
        annotations={"a": ["do a"], "b": ["do b"]},
    )
    assert adapter._mtlc_trials == [("a", 10), ("a", 11), ("b", 20)]
    assert adapter.num_tasks == 3
    assert [adapter.num_init_states(i) for i in range(3)] == [1, 1, 1]
    assert [adapter.task_language(i) for i in range(3)] == ["do a", "do a", "do b"]


def test_lh_official_metrics_and_repeated_task_denominator():
    adapter = object.__new__(CalvinAdapter)
    adapter.cfg = SimpleNamespace(protocol="lh_mtlc")
    adapter.cfg.episode_plan = lambda: [(i, 0, 0) for i in range(8)]
    sequences = [list("abcde") for _ in range(8)]
    lengths = [0, 1, 2, 3, 4, 5, 5, 2]
    metrics = [{"chain_length": length, "subtasks": seq} for length, seq in zip(lengths, sequences)]
    primary, extra = adapter.aggregate(metrics)
    assert primary == {"name": "avg_sequence_length", "value": 2.75, "max": 5.0}
    assert [extra[f"SR{i}"] for i in range(1, 6)] == [0.875, 0.75, 0.5, 0.375, 0.25]
    assert extra["per_task"] == {
        "a": {"success": 7, "total": 8},
        "b": {"success": 6, "total": 7},
        "c": {"success": 4, "total": 6},
        "d": {"success": 3, "total": 4},
        "e": {"success": 2, "total": 3},
    }
    assert extra["success_rate"] == 0.25

    # If a name repeats, the first failed *position* still counts as an
    # attempt even when the same task name succeeded earlier in the chain.
    _, duplicate = adapter.aggregate([
        {"chain_length": 2, "subtasks": ["a", "b", "a", "c", "d"]}
    ])
    assert duplicate["per_task"]["a"] == {"success": 1, "total": 2}


def test_mtlc_aggregate_is_single_step_success_rate():
    adapter = object.__new__(CalvinAdapter)
    adapter.cfg = SimpleNamespace(protocol="mtlc")
    primary, extra = adapter.aggregate([
        {"chain_length": 1, "subtasks": ["a"]},
        {"chain_length": 0, "subtasks": ["a"]},
        {"chain_length": 1, "subtasks": ["b"]},
    ])
    assert primary == {"name": "success_rate", "value": 2 / 3, "max": 1.0}
    assert extra["per_task"] == {
        "a": {"success": 1, "total": 2},
        "b": {"success": 1, "total": 1},
    }


def test_close_swallows_env_close_error():
    class _BadEnv:
        def close(self):
            raise RuntimeError("boom")

    adapter = object.__new__(CalvinAdapter)
    adapter._env = _BadEnv()
    adapter.close()  # must not raise
    assert adapter._env is None


def test_reproducibility_includes_native_cameras(tmp_path):
    root = _dataset_root(tmp_path)
    hydra = tmp_path / "validation" / ".hydra"
    data = yaml.safe_load((hydra / "merged_config.yaml").read_text(encoding="utf-8"))
    data["cameras"] = {
        "static": {"width": 200, "height": 200},
        "gripper": {"width": 84, "height": 84},
    }
    (hydra / "merged_config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    adapter = CalvinAdapter(
        _lh_cfg(root),
        env=_Env(dim=15),
        sequences=[({}, ["a"] * 5)],
        oracle=_NeverOracle(),
        annotations={"a": ["do a"]},
    )
    assert adapter.reproducibility()["native_cameras"] == {
        "image/main": [200, 200],
        "image/second": [84, 84],
    }


def test_example_dataflow_keeps_policy_command_queue_32():
    data = yaml.safe_load((ROOT / "dataflow.example.yaml").read_text(encoding="utf-8"))
    benchmark, policy = data["nodes"]
    assert benchmark["path"] == "./.venv/bin/python"
    assert policy["inputs"]["policy_command"] == {
        "source": "calvin_benchmark/policy_command",
        "queue_size": 32,
    }
    assert (ROOT / "_core" / "runner.py").is_file()
