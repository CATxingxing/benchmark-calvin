"""One CALVIN runtime serving both official suites (M5).

The node prepares LH-MTLC and MTLC once; each run picks a suite and the
adapter, describe payload, plan validation and outputs all follow it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import yaml
from forge_tool import ToolContext, ToolExecutionKey, ToolRequest

from _core.session import BenchSession, bench_limits, capabilities_payload
from adapters.calvin_adapter import CalvinAdapter
from bench_tool.descriptor import BENCH_ENDPOINT_ID, OP_RUN
from bench_tool.endpoints import BenchSessionEndpoint, normalize_plan
from bench_tool.invocations import InvocationTable
from bench_tool.result_shape import is_full_suite
from config import load_config

from test_review_contract import _Env, _NeverOracle, _dataset_root


def _write(tmp_path, **overrides):
    data = {
        "dataset_path": str(tmp_path),
        "protocol": "mtlc",
        "suite": "mtlc",
        "suites": ["mtlc", "lh_mtlc"],
        "action_mode": "absolute_euler",
        "sequence_count": 2,
        "smoke": True,
        "policy_id": "calvin_policy",
        "checkpoint_path": "unused.ckpt",
        "serve": True,
        "max_total_episodes": 10,
        "max_max_steps": 2000,
        "result_dir": str(tmp_path / "results"),
    }
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    root = _dataset_root(tmp_path)
    monkeypatch.setattr(CalvinAdapter, "_load_task_to_id_dict", lambda self: {"a": [10, 11], "b": [20]})
    monkeypatch.setattr(CalvinAdapter, "_prepare_mtlc_dataset_mapping", lambda self: None)
    cfg = load_config(_write(root))
    adapter = CalvinAdapter(
        cfg,
        env=_Env(),
        sequences=[({}, list("abcab")), ({}, list("babab"))],
        oracle=_NeverOracle(),
        annotations={"a": ["do a"], "b": ["do b"]},
    )
    caps = capabilities_payload(cfg, adapter, limits=bench_limits(cfg))
    return cfg, adapter, caps


def test_config_multi_suite_validation(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.available_suites() == ("mtlc", "lh_mtlc")
    assert (cfg.suite, cfg.protocol, cfg.max_steps) == ("mtlc", "mtlc", 240)
    lh = cfg.with_suite("lh_mtlc")
    assert (lh.suite, lh.protocol, lh.max_steps) == ("lh_mtlc", "lh_mtlc", 1800)
    assert (cfg.suite, cfg.max_steps) == ("mtlc", 240)  # template untouched
    with pytest.raises(ValueError, match="not configured"):
        cfg.with_suite("abc")

    with pytest.raises(ValueError, match="must be listed in suites"):
        load_config(_write(tmp_path, suites=["lh_mtlc"]))
    with pytest.raises(ValueError, match="may only contain"):
        load_config(_write(tmp_path, suites=["mtlc", "abc"]))
    with pytest.raises(ValueError, match="must not repeat"):
        load_config(_write(tmp_path, suites=["mtlc", "mtlc"]))
    # LH in the set keeps the formal 1000-sequence rule even if the default is MTLC.
    with pytest.raises(ValueError, match="sequence_count=1000"):
        load_config(_write(tmp_path, smoke=False))
    # MTLC in the set needs the checkpoint even if the default is LH.
    with pytest.raises(ValueError, match="checkpoint_path"):
        load_config(_write(tmp_path, protocol="lh_mtlc", suite="lh_mtlc", checkpoint_path=None))


def test_legacy_single_suite_config_still_loads(tmp_path):
    cfg = load_config(_write(tmp_path, suites=None, protocol="lh_mtlc", suite="lh_mtlc"))
    assert cfg.available_suites() == ("lh_mtlc",)
    with pytest.raises(ValueError, match="not configured"):
        cfg.with_suite("mtlc")


def test_describe_lists_every_suite_and_restores_adapter_cfg(runtime):
    cfg, adapter, caps = runtime
    assert caps["suite"] == "mtlc" and caps["num_tasks"] == 3
    by_name = {item["suite"]: item for item in caps["suites"]}
    assert by_name["mtlc"]["num_tasks"] == 3
    assert by_name["mtlc"]["defaults"] == {"max_steps": 240}
    assert by_name["lh_mtlc"]["num_tasks"] == 2
    assert by_name["lh_mtlc"]["defaults"] == {"max_steps": 1800}
    assert by_name["mtlc"]["tasks"][2]["language"] == "do b"
    assert by_name["lh_mtlc"]["tasks"][1]["language"] == "do b -> do a -> do b -> do a -> do b"
    assert adapter.cfg is cfg


def test_normalize_plan_selects_suite(runtime):
    _, _, caps = runtime
    plan = normalize_plan({"suite": "lh_mtlc", "num_runs": 1}, caps)
    assert plan["suite"] == "lh_mtlc"
    assert plan["task_ids"] == [0, 1]
    assert plan["max_steps"] == 1800
    default = normalize_plan({}, caps)
    assert default["suite"] == "mtlc" and default["task_ids"] == [0, 1, 2] and default["max_steps"] == 240
    with pytest.raises(Exception, match="未知 suite"):
        normalize_plan({"suite": "abc"}, caps)
    # MTLC has 3 trials; task 2 does not exist in LH.
    with pytest.raises(Exception, match="越界 task_ids"):
        normalize_plan({"suite": "lh_mtlc", "task_ids": [2]}, caps)


class _Node:
    def send_output(self, *_args, **_kwargs):
        pass


def test_session_switches_suite_without_touching_template(runtime):
    cfg, adapter, caps = runtime
    plan = normalize_plan({"suite": "lh_mtlc", "task_ids": [1]}, caps)
    session = BenchSession(_Node(), cfg, adapter, plan, "inv-lh", log=lambda _m: None)
    assert session.cfg.suite == session.cfg.protocol == "lh_mtlc"
    assert adapter.cfg is session.cfg
    assert adapter.num_tasks == 2
    assert adapter.task_language(1) == "do b -> do a -> do b -> do a -> do b"
    assert cfg.suite == "mtlc" and cfg.max_steps == 240

    back = BenchSession(_Node(), cfg, adapter, normalize_plan({}, caps), "inv-mtlc", log=lambda _m: None)
    assert back.cfg.protocol == "mtlc" and adapter.num_tasks == 3


def test_rejected_session_rolls_back_adapter_cfg(runtime):
    cfg, adapter, _ = runtime
    bad = {"suite": "lh_mtlc", "task_ids": [2], "init_state_ids": [0], "num_runs": 1, "seed": 0, "max_steps": 1800}
    with pytest.raises(ValueError):
        BenchSession(_Node(), cfg, adapter, bad, "inv-bad", log=lambda _m: None)
    assert adapter.cfg is cfg and adapter.num_tasks == 3


def test_is_full_suite_requires_full_coverage(runtime):
    _, _, caps = runtime
    full = normalize_plan({"suite": "lh_mtlc"}, caps)
    assert is_full_suite(plan=full, capabilities=caps, completed_episodes=2, status="succeeded")
    assert not is_full_suite(plan=full, capabilities=caps, completed_episodes=1, status="succeeded")
    assert not is_full_suite(plan=full, capabilities=caps, completed_episodes=2, status="stopped")
    part = normalize_plan({"suite": "mtlc", "task_ids": [0, 1]}, caps)
    assert not is_full_suite(plan=part, capabilities=caps, completed_episodes=2, status="succeeded")


def test_endpoint_rejects_unknown_suite_synchronously(runtime):
    cfg, adapter, caps = runtime
    endpoint = BenchSessionEndpoint(InvocationTable(), _Node(), cfg, adapter, caps, emit=lambda *_: None, log=lambda _m: None)
    key = ToolExecutionKey(invocation_id="inv-x", attempt_id="a1")
    context = ToolContext(
        execution_key=key,
        tool_id="calvin.benchmark.run",
        implementation_id="calvin-benchmark.benchmark",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=OP_RUN,
    )
    with pytest.raises(Exception, match="未知 suite"):
        asyncio.run(endpoint.start(ToolRequest(arguments={"suite": "abc"}), context, events=None))
    assert endpoint.take_pending() == []
