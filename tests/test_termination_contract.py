"""Runner-level termination contract tests for the CALVIN timeout fix.

These lock the adapter -> shared-runner coupling: the adapter publishes the
current sub-task deadline via ``cfg.max_steps`` so the frozen runner reports
``timed_out`` at the exact official budget boundary, while scored quantities
(``chain_length``, ``num_steps``, ``success``) stay unchanged.

They are self-contained: the only CALVIN import touched by ``reset_episode``
(``get_env_state_for_initial_condition``) is replaced by a shim, so the suite
still runs without a CALVIN checkout.
"""
from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from forge_msgs import JointCommand

from adapters.calvin_adapter import CalvinAdapter

_STATE = {
    "led": 0, "lightbulb": 0, "slider": "left", "drawer": "closed",
    "red_block": "table", "blue_block": "table", "pink_block": "table", "grasped": 0,
}


def _runner_mod():
    return importlib.import_module("_core.runner")


def _config_mod():
    return importlib.import_module("_core.config")


@pytest.fixture
def fake_calvin(monkeypatch):
    pkg = types.ModuleType("calvin_agent")
    pkg.__path__ = []
    ev = types.ModuleType("calvin_agent.evaluation")
    ev.__path__ = []
    utils = types.ModuleType("calvin_agent.evaluation.utils")
    utils.get_env_state_for_initial_condition = lambda _state: (np.zeros(15), np.zeros(24))
    monkeypatch.setitem(sys.modules, "calvin_agent", pkg)
    monkeypatch.setitem(sys.modules, "calvin_agent.evaluation", ev)
    monkeypatch.setitem(sys.modules, "calvin_agent.evaluation.utils", utils)
    pkg.evaluation = ev
    ev.utils = utils
    yield


class _Env:
    def __init__(self):
        self.closed = False

    def _obs(self):
        return {
            "robot_obs": np.zeros(15),
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
        return self._obs(), 0.0, False, {}

    def close(self):
        self.closed = True


class _Oracle:
    """Solves the oracle at the given global call numbers."""

    def __init__(self, solve_calls=()):
        self.calls = 0
        self.solve_calls = set(solve_calls)

    def get_task_info_for_set(self, *_args):
        self.calls += 1
        return {"ok"} if self.calls in self.solve_calls else set()


class _Node:
    def __init__(self):
        self.outputs = []

    def send_output(self, name, value):
        self.outputs.append(name)


def _dataset_root(tmp_path):
    hydra = tmp_path / "validation" / ".hydra"
    hydra.mkdir(parents=True, exist_ok=True)
    (hydra / "merged_config.yaml").write_text(
        yaml.safe_dump(
            {
                "robot": {"euler_obs": True, "arm_joint_ids": list(range(7))},
                "cameras": {
                    "static": {"width": 4, "height": 5},
                    "gripper": {"width": 2, "height": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _make_lh_adapter(tmp_path, solve_calls):
    cfg = SimpleNamespace(
        dataset_path=str(_dataset_root(tmp_path)),
        action_mode="absolute_euler",
        protocol="lh_mtlc",
        sequence_count=1,
    )
    adapter = CalvinAdapter(
        cfg,
        env=_Env(),
        sequences=[(dict(_STATE), ["a", "b", "c", "d", "e"])],
        oracle=_Oracle(solve_calls),
        annotations={t: [f"do {t}"] for t in "abcde"},
    )
    return cfg, adapter


def _run(cfg, adapter, limit=4000):
    config_mod = _config_mod()
    runner_mod = _runner_mod()
    base = config_mod.BaseBenchmarkConfig(
        benchmark="calvin", suite="lh_mtlc", task_ids=[0], init_state_ids=[0],
        num_runs=1, seed=0, policy_id="fake", result_dir=None, record_dir=None,
    )
    base.protocol = cfg.protocol
    base.dataset_path = cfg.dataset_path
    adapter.cfg = base
    runner = runner_mod.BenchmarkRunner(_Node(), base, adapter, log=lambda _m: None)
    runner.result.dump = lambda *a, **k: None
    runner.advance()
    cmd = JointCommand(name=[f"a{i}" for i in range(7)], mode="position", position=[0.0] * 7)
    for _ in range(limit):
        if runner.finished:
            break
        runner.on_action(cmd)
    return runner.result.episodes[0]


def test_lh_subtask_timeout_is_timed_out(tmp_path, fake_calvin):
    cfg, adapter = _make_lh_adapter(tmp_path, solve_calls=())
    ep = _run(cfg, adapter)
    assert ep.termination == "timed_out"
    assert ep.success is False
    assert ep.num_steps == 360            # official per-sub-task budget
    assert ep.metrics["chain_length"] == 0


def test_lh_timeout_after_one_solve_reports_460_steps(tmp_path, fake_calvin):
    cfg, adapter = _make_lh_adapter(tmp_path, solve_calls={100})
    ep = _run(cfg, adapter)
    assert ep.termination == "timed_out"
    assert ep.num_steps == 460            # 100 (solve) + 360 (next sub-task budget)
    assert ep.metrics["chain_length"] == 1


def test_lh_full_chain_still_success(tmp_path, fake_calvin):
    cfg, adapter = _make_lh_adapter(tmp_path, solve_calls={10, 20, 30, 40, 50})
    ep = _run(cfg, adapter)
    assert ep.termination == "success"
    assert ep.success is True
    assert ep.num_steps == 50
    assert ep.metrics["chain_length"] == 5


def _make_mtlc_adapter(tmp_path, monkeypatch, solve_calls):
    root = _dataset_root(tmp_path)
    frame = tmp_path / "episode_0000000.npz"
    np.savez(frame, robot_obs=np.zeros(15), scene_obs=np.zeros(24))
    monkeypatch.setattr(CalvinAdapter, "_load_task_to_id_dict", lambda self: {"a": [0]})
    monkeypatch.setattr(CalvinAdapter, "_prepare_mtlc_dataset_mapping", lambda self: None)
    monkeypatch.setattr(CalvinAdapter, "_mtlc_frame_file", lambda self, _i: frame)
    cfg = SimpleNamespace(
        dataset_path=str(root),
        action_mode="absolute_euler",
        protocol="mtlc",
        checkpoint_path="unused",
    )
    adapter = CalvinAdapter(
        cfg, env=_Env(), oracle=_Oracle(solve_calls), annotations={"a": ["do a"]},
    )
    return cfg, adapter


def test_mtlc_budget_timeout_is_timed_out(tmp_path, monkeypatch):
    cfg, adapter = _make_mtlc_adapter(tmp_path, monkeypatch, solve_calls=())
    ep = _run(cfg, adapter, limit=2000)
    assert ep.termination == "timed_out"
    assert ep.success is False
    assert ep.num_steps == 240            # official MTLC budget
    assert ep.metrics["chain_length"] == 0


def test_mtlc_success_still_success(tmp_path, monkeypatch):
    cfg, adapter = _make_mtlc_adapter(tmp_path, monkeypatch, solve_calls={5})
    ep = _run(cfg, adapter, limit=2000)
    assert ep.termination == "success"
    assert ep.success is True
    assert ep.num_steps == 5


def test_lh_cfg_max_steps_tracks_subtask_deadline(tmp_path, fake_calvin):
    cfg, adapter = _make_lh_adapter(tmp_path, solve_calls={100})
    base = _config_mod().BaseBenchmarkConfig(
        benchmark="calvin", suite="lh_mtlc", task_ids=[0], init_state_ids=[0],
        num_runs=1, seed=0, policy_id="fake", result_dir=None, record_dir=None,
        max_steps=1800,
    )
    base.protocol = "lh_mtlc"
    adapter.cfg = base
    adapter.reset_episode(task_id=0, init_state_id=0, run_index=0, episode_seed=0)
    adapter.step(np.zeros(7))
    assert adapter.cfg.max_steps == 360
    for _ in range(99):
        adapter.step(np.zeros(7))
    # subtask 1 solved on the 100th action -> next deadline is 100 + 360.
    assert adapter.cfg.max_steps == 460
