"""CALVIN node configuration."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any


def _import_core(name: str):
    import importlib

    module = importlib.import_module("_core." + name)
    actual = Path(module.__file__).resolve()
    expected = Path(__file__).resolve().parent / "_core" / f"{name}.py"
    if actual != expected:
        raise RuntimeError(f"benchmark-calvin must use its vendored _core: {actual}")
    return module

_core = _import_core("config")
BaseBenchmarkConfig = _core.BaseBenchmarkConfig
fill_base = _core.fill_base
read_yaml = _core.read_yaml


@dataclass
class CalvinConfig(BaseBenchmarkConfig):
    benchmark: str = "calvin"
    suite: str = ""
    dataset_path: str = ""
    protocol: str = "lh_mtlc"
    action_mode: str | None = None
    sequence_count: int = 1000
    checkpoint_path: str | None = None
    policy_id: str = ""
    control_mode: str = ""


def _strict_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{field} must be >= 1")
    return result


def _validate_id_spec(value: Any, *, field: str) -> None:
    if value is None or value == "all":
        return
    if isinstance(value, bool):
        raise ValueError(f"{field} must not contain booleans")
    if isinstance(value, Integral):
        return
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
            raise ValueError(f"{field} list must contain integers only")
        return
    if isinstance(value, str):
        # The shared Core supports comma/range syntax. Validate characters here
        # so an accidental float or arbitrary string is not silently int-cast.
        allowed = set("0123456789,- ")
        if value and set(value) <= allowed:
            return
    raise ValueError(f"unsupported {field} specification: {value!r}")


def load_config(path: str | Path) -> CalvinConfig:
    data = read_yaml(path)

    dataset_path = data.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError("dataset_path is required and must be the CALVIN dataset root")
    protocol = data.get("protocol", "lh_mtlc")
    if not isinstance(protocol, str):
        raise ValueError("protocol must be a string")
    protocol = protocol.lower()
    action_mode = data.get("action_mode")
    if not isinstance(action_mode, str) or not action_mode:
        raise ValueError("action_mode is required; no CALVIN action semantic is guessed")
    raw_num_runs = data.get("num_runs", 1)
    if isinstance(raw_num_runs, bool) or not isinstance(raw_num_runs, Integral) or int(raw_num_runs) != 1:
        raise ValueError("official CALVIN evaluation requires num_runs=1")
    raw_send_instruction = data.get("send_instruction", True)
    if raw_send_instruction is not True:
        raise ValueError("official CALVIN evaluation requires send_instruction=true")
    _validate_id_spec(data.get("task_ids"), field="task_ids")
    _validate_id_spec(data.get("init_state_ids"), field="init_state_ids")

    cfg = CalvinConfig()
    fill_base(cfg, data)
    cfg.dataset_path = dataset_path
    cfg.protocol = protocol
    cfg.action_mode = action_mode
    cfg.sequence_count = _strict_positive_int(data.get("sequence_count", cfg.sequence_count), field="sequence_count")
    cfg.checkpoint_path = data.get("checkpoint_path", cfg.checkpoint_path)

    # Official evaluator inputs are intentionally not configurable here.
    # Language annotations and TaskOracle configuration are loaded from the
    # actual imported CALVIN checkout; MTLC task ids must come from checkpoint.
    forbidden = {"annotation_path", "task_config_path", "task_to_id_dict", "euler_obs"} & set(data)
    if forbidden:
        raise ValueError(
            "official CALVIN evaluator inputs cannot be overridden from benchmark config: "
            + ", ".join(sorted(forbidden))
        )

    if cfg.protocol not in {"lh_mtlc", "mtlc"}:
        raise ValueError("protocol must be 'lh_mtlc' or 'mtlc'")
    if "suite" not in data or not isinstance(data["suite"], str) or not data["suite"].strip():
        raise ValueError("suite is required and must explicitly match protocol")
    if cfg.suite != cfg.protocol:
        raise ValueError(f"suite must match protocol ({cfg.protocol!r}), got {data['suite']!r}")
    if not isinstance(data.get("policy_id"), str) or not data["policy_id"].strip():
        raise ValueError("policy_id is required and must match the target Forge Policy config")
    cfg.policy_id = data["policy_id"].strip()
    cfg.control_mode = cfg.action_mode
    if cfg.protocol == "mtlc":
        if not isinstance(cfg.checkpoint_path, str) or not cfg.checkpoint_path:
            raise ValueError("MTLC requires checkpoint_path containing task_to_id_dict")

    # Formal LH-MTLC always uses the official 1000-sequence set. A shorter
    # sequence set is permitted only for an explicitly labelled smoke run.
    if cfg.protocol == "lh_mtlc" and cfg.sequence_count != 1000 and data.get("smoke") is not True:
        raise ValueError("formal LH-MTLC requires sequence_count=1000 (set smoke: true only for a short smoke)")

    # The shared Runner owns an episode-wide limit; protocol code separately
    # enforces CALVIN's exact per-subtask budget.
    cfg.max_steps = 1800 if cfg.protocol == "lh_mtlc" else 240
    return cfg


BenchmarkNodeConfig = CalvinConfig
