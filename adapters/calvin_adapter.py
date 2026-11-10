"""CALVIN -> Forge adapter.

Only CALVIN-native reset, observation, action and TaskOracle semantics live in
this module.  Policy preprocessing/chunking/recurrent state remains entirely in
the unmodified Forge Policy runtime.

Layout (single file, ordered by responsibility):
    1. runtime bootstrap + compatibility shims
    2. action codec + official protocol state machines (LH-MTLC / MTLC)
    3. official dataset-index helpers
    4. ``CalvinAdapter`` (the ``_core`` BenchmarkAdapter implementation)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _core.adapter import StepOutcome

# ---------------------------------------------------------------------------
# Constants fixed by the CALVIN official evaluation protocol.
# ---------------------------------------------------------------------------

LH_MAX_ACTIONS = 360  # official ``evaluate_policy.EP_LEN`` per subtask
MTLC_MAX_ACTIONS = 240  # official ``evaluate_policy_singlestep`` ep_len
OBSERVATION_SEMANTICS = (
    "CALVIN native HWC RGB; no flip/resize/normalize; integer 0..255 carrier -> uint8"
)


# ===========================================================================
# 1. Runtime bootstrap + compatibility shims
# ===========================================================================


def _configure_calvin_import_paths() -> None:
    """Expose the CALVIN checkout (``calvin_env`` / ``calvin_models`` / bundled ``tacto``).

    Executed at adapter-module import time. All third-party packages come from the
    Node's private uv environment; only the upstream CALVIN source tree is added,
    from ``CALVIN_ROOT`` (``~`` is expanded because dataflow env values are not).
    """
    calvin_root = os.environ.get("CALVIN_ROOT")
    if not calvin_root:
        return
    base = Path(calvin_root).expanduser().resolve()
    for path in (base / "calvin_env", base / "calvin_models", base / "calvin_env" / "tacto"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _apply_numpy_compat() -> None:
    """Process-local alias for bundled tacto 0.0.3 on modern NumPy.

    tacto references the removed ``np.float``; aliasing it here leaves the
    CALVIN checkout untouched and changes no observation data.
    """
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]


_configure_calvin_import_paths()


# ===========================================================================
# 2. Action codec + official protocol state machines
# ===========================================================================


@dataclass(frozen=True)
class CalvinActionCodec:
    """Validate and encode a Forge flat action for the native CALVIN env.

    ``mode`` is deliberately required.  ``relative_euler`` leaves CALVIN's
    native scaling in :meth:`Robot.relative_to_absolute`; absolute modes are
    split into the official ``[position, orientation, gripper]`` container.
    Quaternion actions use CALVIN/pybullet's native ``(x, y, z, w)`` order.
    """

    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {"relative_euler", "absolute_euler", "absolute_quaternion"}:
            raise ValueError(
                "action_mode must be one of relative_euler, absolute_euler, "
                f"absolute_quaternion; got {self.mode!r}"
            )

    @property
    def dimension(self) -> int:
        return 8 if self.mode == "absolute_quaternion" else 7

    def encode(self, action: Any) -> np.ndarray | list[np.ndarray]:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != self.dimension:
            raise ValueError(
                f"{self.mode} requires exactly {self.dimension} values, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("CALVIN action contains non-finite values")

        # CALVIN's wrapper maps any positive gripper command to +1 and all
        # other values to -1.  Keep this explicit and identical for absolute
        # and relative actions.
        values = values.copy()
        values[-1] = 1.0 if values[-1] > 0 else -1.0
        if self.mode == "relative_euler":
            return values
        orientation_end = 6 if self.mode == "absolute_euler" else 7
        # Keep gripper as a length-one array, exactly as CALVIN's
        # ``np.split(..., [3, 6])`` wrapper does; Robot.apply_action accepts
        # that container and normalises it to an integer.
        return [values[:3].copy(), values[3:orientation_end].copy(), values[-1:].copy()]


@dataclass
class LhMtlcProtocol:
    """Small state machine mirroring ``evaluate_sequence``/``rollout``.

    The adapter owns the environment and calls this object after each native
    step.  A sequence has one reset, a fresh ``start_info`` for every subtask,
    and a hard ``max_actions`` budget per subtask (official 360).
    """

    sequence: tuple[str, ...]
    language: Callable[[str], str]
    max_actions: int = LH_MAX_ACTIONS
    index: int = 0
    actions: int = 0
    completed: int = 0
    start_info: dict[str, Any] | None = None

    def begin(self, info: dict[str, Any]) -> None:
        self.index = 0
        self.actions = 0
        self.completed = 0
        self.start_info = info

    @property
    def current_task(self) -> str:
        if not self.sequence or self.index >= len(self.sequence):
            raise RuntimeError("LH-MTLC protocol has no current subtask")
        return self.sequence[self.index]

    @property
    def current_instruction(self) -> str:
        return self.language(self.current_task)

    def observe(self, current_info: dict[str, Any], oracle: Any) -> dict[str, Any]:
        if self.start_info is None:
            raise RuntimeError("protocol.begin() must precede observe()")
        self.actions += 1
        task = self.current_task
        solved = bool(oracle.get_task_info_for_set(self.start_info, current_info, {task}))
        if solved:
            self.completed += 1
            self.index += 1
            if self.index == len(self.sequence):
                return {
                    "success": True,
                    "done": True,
                    "chain_length": self.completed,
                    "subtask": task,
                    "subtasks_completed": self.completed,
                }
            self.start_info = current_info
            self.actions = 0
            return {
                "success": False,
                "done": False,
                "next_instruction": self.current_instruction,
                "chain_length": self.completed,
                "subtasks_completed": self.completed,
            }

        if self.actions >= self.max_actions:
            return {
                "success": False,
                "done": True,
                "chain_length": self.completed,
                "failed_subtask": task,
                "subtasks_completed": self.completed,
                "timed_out_subtask": True,
            }
        return {
            "success": False,
            "done": False,
            "chain_length": self.completed,
            "subtasks_completed": self.completed,
        }


def _dataset_index(value: Any) -> int:
    """Accept only integer-like checkpoint indices; never truncate floats."""
    if isinstance(value, bool):
        raise ValueError("MTLC dataset indices must be integers, not bool")
    if isinstance(value, Integral):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if isinstance(scalar, Integral) and not isinstance(scalar, bool):
            return int(scalar)
    raise ValueError(f"MTLC dataset index must be integer-like, got {value!r}")


@dataclass
class MtlcProtocol:
    """One-task MTLC budget, mirroring :class:`LhMtlcProtocol` for one subtask.

    Reset is once per trial, ``start_info`` is captured at reset, and the action
    budget is the official 240.  ``task_to_id_dict`` comes from the checkpoint.
    """

    task_to_id_dict: Mapping[str, Any]
    max_actions: int = MTLC_MAX_ACTIONS
    # per-episode state (reset by ``begin``)
    task: str | None = None
    start_info: dict[str, Any] | None = None
    steps: int = 0

    def __post_init__(self) -> None:
        if not self.task_to_id_dict:
            raise ValueError("MTLC requires a non-empty checkpoint task_to_id_dict")

    def episode_ids(self, task: str) -> tuple[int, ...]:
        if task not in self.task_to_id_dict:
            raise KeyError(f"task {task!r} is absent from checkpoint task_to_id_dict")
        values = self.task_to_id_dict[task]
        if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
            raise ValueError(f"checkpoint task_to_id_dict[{task!r}] must be an iterable of dataset ids")
        result = tuple(_dataset_index(value) for value in values)
        if not result:
            raise ValueError(f"checkpoint task_to_id_dict[{task!r}] is empty")
        if any(value < 0 for value in result):
            raise ValueError(f"checkpoint task_to_id_dict[{task!r}] contains a negative dataset id")
        return result

    def begin(self, task: str, start_info: dict[str, Any]) -> None:
        self.task = task
        self.start_info = start_info
        self.steps = 0

    def observe(self, current_info: dict[str, Any], oracle: Any) -> dict[str, Any]:
        if self.task is None or self.start_info is None:
            raise RuntimeError("protocol.begin() must precede observe()")
        solved = bool(oracle.get_task_info_for_set(self.start_info, current_info, {self.task}))
        self.steps += 1
        done = solved or self.steps >= self.max_actions
        result: dict[str, Any] = {
            "success": solved,
            "done": done,
            "steps": self.steps,
            "chain_length": int(solved),
            "subtasks": [self.task],
        }
        if not solved and done:
            result["failed_subtask"] = self.task
        return result


# ===========================================================================
# 3. Official dataset-index helpers
# ===========================================================================


def _episode_lookup_from_bounds(bounds: np.ndarray, min_window: int) -> np.ndarray:
    """Exact ``DiskDataset._build_file_indices`` mapping for validation data."""
    array = np.asarray(bounds)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"ep_start_end_ids.npy must have shape (N,2), got {array.shape}")
    lookup: list[int] = []
    for start, end in array:
        start_i, end_i = int(start), int(end)
        if start_i < 0 or end_i < start_i:
            raise ValueError(f"invalid validation episode bounds: {(start_i, end_i)}")
        lookup.extend(range(start_i, end_i + 1 - min_window))
    return np.asarray(lookup, dtype=np.int64)


# ===========================================================================
# 4. CalvinAdapter (the _core BenchmarkAdapter implementation)
# ===========================================================================


class CalvinAdapter:
    SUCCESS_SOURCE = "calvin_env.envs.tasks.Tasks.get_task_info_for_set"
    RECORD_FPS = 30.0

    # Cameras this adapter actually publishes (to_images maps exactly these).
    # The official evaluator never builds the full camera set: it constructs the
    # env through CalvinEnvWrapper, which forwards the checkpoint observation
    # space so that unused cameras (notably the tactile sensor) are removed from
    # the merged config before instantiation.  Mirroring that here keeps env
    # construction in the same scope as upstream and avoids importing the
    # tactile renderer's separate dependency chain.  It changes no rendered
    # pixel, no reset state and no TaskOracle input.
    CAMERA_OBS_SPACE: dict[str, list[str]] = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
    }

    def __init__(
        self,
        cfg: Any,
        *,
        env: Any | None = None,
        env_factory: Callable[[Path], Any] | None = None,
        sequences: list[tuple[Mapping[str, Any], list[str]]] | None = None,
        oracle: Any | None = None,
        annotations: Mapping[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_path).expanduser().resolve()
        self.validation_path = self.dataset_root / "validation"
        if not self.validation_path.is_dir() and env is None:
            raise FileNotFoundError(f"CALVIN validation directory not found: {self.validation_path}")

        self.codec = CalvinActionCodec(str(cfg.action_mode))
        self._env = env
        self._env_factory = env_factory
        self._oracle = oracle
        self._annotations = dict(annotations or {})
        self._protocol: LhMtlcProtocol | None = None
        self._mtlc_protocol: MtlcProtocol | None = None
        self._task_id: int | None = None
        self._obs: dict[str, Any] | None = None
        self._episode_metrics: dict[str, Any] = {}
        # Cumulative action counters.  They let the adapter publish the current
        # official sub-task deadline through the shared runner's ``max_steps``
        # so budget exhaustion is reported as ``timed_out`` (manual §3.4 rule 3);
        # see ``_lh_step`` / ``_mtlc_step``.
        self._lh_actions = 0
        self._mtlc_actions = 0

        self._mtlc_trials: list[tuple[str, int]] = []
        self._mtlc_episode_lookup: np.ndarray | None = None
        self._mtlc_name_pattern: tuple[Path, str] | None = None
        self._mtlc_n_digits: int | None = None

        # Every configured suite is prepared once so a single runtime can run
        # either protocol; ``_sequences`` then follows ``self.cfg.protocol``.
        self._suite_sequences: dict[str, list[tuple[Mapping[str, Any], list[str]]]] = {}
        suites = tuple(getattr(cfg, "suites", None) or (cfg.protocol,))
        if "mtlc" in suites:
            task_map = self._load_task_to_id_dict()
            self._mtlc_protocol = MtlcProtocol(task_map)
            for task in task_map:
                self._mtlc_trials.extend(
                    (str(task), dataset_id) for dataset_id in self._mtlc_protocol.episode_ids(str(task))
                )
            if not self._mtlc_trials:
                raise ValueError("MTLC checkpoint task_to_id_dict produced no evaluation trials")
            self._suite_sequences["mtlc"] = [({}, [task]) for task, _ in self._mtlc_trials]
            self._prepare_mtlc_dataset_mapping()
        if "lh_mtlc" in suites:
            if sequences is None:
                from calvin_agent.evaluation.multistep_sequences import get_sequences

                sequences = list(get_sequences(int(cfg.sequence_count)))
            lh_sequences = [(state, list(sequence)) for state, sequence in sequences]
            if len(lh_sequences) != int(cfg.sequence_count):
                raise RuntimeError("get_sequences returned an unexpected sequence count")
            if any(len(sequence) != 5 for _, sequence in lh_sequences):
                raise RuntimeError("every CALVIN LH-MTLC sequence must contain exactly five subtasks")
            self._suite_sequences["lh_mtlc"] = lh_sequences
        if cfg.protocol not in self._suite_sequences:
            raise ValueError(f"default protocol {cfg.protocol!r} is not among prepared suites {list(suites)}")

        if self._oracle is None:
            self._oracle = self._load_task_oracle()
        self._state_names = self._resolve_state_names()
        self._native_cameras = self._resolve_native_cameras()
        self._log_observation_contract()

    @property
    def _sequences(self) -> list[tuple[Mapping[str, Any], list[str]]]:
        prepared = self.__dict__.get("_suite_sequences") or {}
        protocol = self.cfg.protocol
        if protocol not in prepared:
            raise ValueError(f"suite {protocol!r} was not prepared by this adapter; prepared: {list(prepared)}")
        return prepared[protocol]

    def available_suites(self) -> tuple[str, ...]:
        return tuple(self.__dict__.get("_suite_sequences") or ())

    # ---- official CALVIN sources ------------------------------------------

    @staticmethod
    def _calvin_conf_root() -> Path:
        import calvin_agent

        package = Path(calvin_agent.__file__).resolve()
        conf = package.parents[1] / "conf"
        if not conf.is_dir():
            raise FileNotFoundError(f"CALVIN conf directory not found next to imported calvin_agent: {conf}")
        return conf

    def _load_task_oracle(self) -> Any:
        import hydra
        from omegaconf import OmegaConf

        path = self._calvin_conf_root() / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"official CALVIN TaskOracle config not found: {path}")
        return hydra.utils.instantiate(OmegaConf.load(path))

    def _load_annotations(self) -> Mapping[str, Any]:
        if self._annotations:
            return self._annotations
        from omegaconf import OmegaConf

        path = self._calvin_conf_root() / "annotations" / "new_playtable_validation.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"official CALVIN validation annotations not found: {path}")
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(loaded, Mapping):
            raise TypeError("official CALVIN validation annotations must be a mapping")
        self._annotations = dict(loaded)
        return self._annotations

    def _language(self, task: str) -> str:
        value = self._load_annotations().get(task)
        if value is None:
            raise KeyError(f"CALVIN language annotation missing for task {task!r}")
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"CALVIN language annotation for {task!r} must be a non-empty sequence")
        # Official evaluator uses val_annotations[subtask][0].
        return str(value[0])

    # ---- observation contract (state names, cameras, startup self-check) --

    def _merged_config(self) -> dict[str, Any]:
        merged = self.validation_path / ".hydra" / "merged_config.yaml"
        if not merged.is_file():
            raise FileNotFoundError(f"CALVIN validation merged_config.yaml not found: {merged}")
        import yaml

        return yaml.safe_load(merged.read_text(encoding="utf-8")) or {}

    def _resolve_state_names(self) -> list[str]:
        # Raw robot_obs is generated by calvin_env.robot.Robot.  Its labels are
        # determined by robot.euler_obs and robot.arm_joint_ids in the dataset's
        # exact merged simulation config; do not guess them from a checkpoint.
        data = self._merged_config()
        try:
            robot = data["robot"]
            euler = robot["euler_obs"]
            arm_joint_ids = robot["arm_joint_ids"]
        except (KeyError, TypeError) as exc:
            raise ValueError("CALVIN merged config must declare robot.euler_obs and robot.arm_joint_ids") from exc
        if not isinstance(euler, bool):
            raise ValueError(f"CALVIN robot.euler_obs must be boolean, got {euler!r}")
        if not isinstance(arm_joint_ids, list) or not arm_joint_ids or any(
            isinstance(value, bool) or not isinstance(value, int) for value in arm_joint_ids
        ):
            raise ValueError("CALVIN robot.arm_joint_ids must be a non-empty integer list")

        orientation = [f"tcp_orn_{axis}" for axis in (("x", "y", "z") if euler else ("x", "y", "z", "w"))]
        return [
            "tcp_pos_x",
            "tcp_pos_y",
            "tcp_pos_z",
            *orientation,
            "gripper_opening_width",
            *[f"robot_joint_{joint_id}" for joint_id in arm_joint_ids],
            "gripper_action",
        ]

    def _resolve_native_cameras(self) -> dict[str, list[int]]:
        # Native render resolution this node publishes, keyed by dora channel.
        # Scaling is the policy's job (manual §3.2); logging it at startup makes
        # a mis-set resolution visible instead of silently dropping the score.
        try:
            data = self._merged_config()
        except FileNotFoundError:
            return {}
        cameras = data.get("cameras")
        if not isinstance(cameras, dict):
            return {}
        result: dict[str, list[int]] = {}
        for key, alias in (("static", "image/main"), ("gripper", "image/second")):
            cam = cameras.get(key)
            if (
                isinstance(cam, dict)
                and isinstance(cam.get("width"), int)
                and isinstance(cam.get("height"), int)
            ):
                result[alias] = [int(cam["width"]), int(cam["height"])]
        return result

    def _observation_contract(self) -> dict[str, Any]:
        """Single source of truth for the published observation contract."""
        return {
            "protocol": self.cfg.protocol,
            "action_mode": self.codec.mode,
            "native_cameras": getattr(self, "_native_cameras", {}),
            "observation_semantics": OBSERVATION_SEMANTICS,
        }

    def _log_observation_contract(self) -> None:
        # Startup self-check (manual §3.2 item 5 / §8): resolution layering,
        # camera aliases and proprio/observation semantics.  This node logs what
        # it renders and publishes; the target checkpoint's expected input size
        # is logged by the policy node.
        contract = self._observation_contract()
        print(
            "[calvin_adapter] observation contract: "
            f"protocol={contract['protocol']} action_mode={contract['action_mode']} "
            f"native_cameras={contract['native_cameras']} "
            f"observation_semantics={contract['observation_semantics']}",
            flush=True,
        )

    # ---- MTLC official dataset-index mapping ------------------------------

    def _load_task_to_id_dict(self) -> Mapping[str, Any]:
        checkpoint_path = getattr(self.cfg, "checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError("MTLC requires checkpoint_path; no fallback task sampling is allowed")
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MTLC checkpoint not found: {path}")
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("MTLC checkpoint must load to a mapping")
        value = checkpoint.get("task_to_id_dict")
        if not isinstance(value, Mapping) or not value:
            raise ValueError("MTLC checkpoint is missing non-empty task_to_id_dict")
        return value

    def _prepare_mtlc_dataset_mapping(self) -> None:
        """Mirror DiskDataset's dataset-index -> first raw frame mapping.

        Official ``evaluate_policy_singlestep.py`` does ``episode = dataset[i]``
        and then resets from ``episode['state_info'][...][0]``.  DiskDataset
        first maps ``i`` through ``episode_lookup``; treating ``i`` as the raw
        ``episode_XXXX.npz`` suffix would reset to the wrong state.
        """
        import yaml
        from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern

        vision_cfg_path = self._calvin_conf_root() / "datamodule" / "datasets" / "vision_dataset" / "vision.yaml"
        if not vision_cfg_path.is_file():
            raise FileNotFoundError(f"official CALVIN vision dataset config not found: {vision_cfg_path}")
        vision_cfg = yaml.safe_load(vision_cfg_path.read_text(encoding="utf-8")) or {}
        min_window = vision_cfg.get("min_window_size")
        save_format = vision_cfg.get("save_format")
        if isinstance(min_window, bool) or not isinstance(min_window, int) or min_window < 1:
            raise ValueError("official CALVIN vision min_window_size is invalid")
        if save_format != "npz":
            raise ValueError(f"official CALVIN MTLC adapter currently requires save_format='npz', got {save_format!r}")

        bounds_path = self.validation_path / "ep_start_end_ids.npy"
        if not bounds_path.is_file():
            raise FileNotFoundError(f"CALVIN validation ep_start_end_ids.npy not found: {bounds_path}")
        bounds = np.load(bounds_path, allow_pickle=False)
        self._mtlc_episode_lookup = _episode_lookup_from_bounds(bounds, min_window)
        self._mtlc_name_pattern, self._mtlc_n_digits = lookup_naming_pattern(self.validation_path, save_format)

        if not len(self._mtlc_episode_lookup):
            raise ValueError("CALVIN validation dataset produced an empty MTLC episode_lookup")
        for _, dataset_id in self._mtlc_trials:
            if not 0 <= dataset_id < len(self._mtlc_episode_lookup):
                raise IndexError(
                    f"checkpoint dataset id {dataset_id} is outside validation dataset length "
                    f"{len(self._mtlc_episode_lookup)}"
                )

    def _mtlc_frame_file(self, dataset_id: int) -> Path:
        if self._mtlc_episode_lookup is None or self._mtlc_name_pattern is None or self._mtlc_n_digits is None:
            raise RuntimeError("MTLC dataset mapping is not initialized")
        raw_frame_id = int(self._mtlc_episode_lookup[dataset_id])
        prefix, suffix = self._mtlc_name_pattern
        path = Path(f"{prefix}{raw_frame_id:0{self._mtlc_n_digits}d}{suffix}")
        if not path.is_file():
            raise FileNotFoundError(f"CALVIN validation frame for dataset id {dataset_id} not found: {path}")
        return path

    # ---- BenchmarkAdapter contract ----------------------------------------

    @property
    def num_tasks(self) -> int:
        return len(self._sequences)

    def num_init_states(self, task_id: int) -> int:
        self._check_task(task_id)
        # LH sequence identity already includes its initial condition.  MTLC is
        # flattened to one Forge task per official (task, dataset_id) trial so
        # variable checkpoint list lengths are represented without truncation.
        return 1

    def task_language(self, task_id: int) -> str:
        self._check_task(task_id)
        if self.cfg.protocol == "mtlc":
            return self._language(self._mtlc_trials[task_id][0])
        return " -> ".join(self._language(task) for task in self._sequences[task_id][1])

    def episode_instruction(self, task_id: int, init_state_id: int, run_index: int) -> str:
        self._check_task(task_id)
        if init_state_id != 0:
            raise IndexError("CALVIN adapter exposes exactly init_state_id=0 per Forge task")
        if self.cfg.protocol == "mtlc":
            return self._language(self._mtlc_trials[task_id][0])
        return self._language(self._sequences[task_id][1][0])

    def state_joint_names(self) -> list[str]:
        return list(self._state_names)

    def reproducibility(self) -> dict[str, Any]:
        contract = self._observation_contract()
        value = {
            "dataset_root": str(self.dataset_root),
            "validation_path": str(self.validation_path),
            "protocol": contract["protocol"],
            "action_mode": contract["action_mode"],
            "max_actions_per_subtask": LH_MAX_ACTIONS if self.cfg.protocol == "lh_mtlc" else MTLC_MAX_ACTIONS,
            "native_cameras": contract["native_cameras"],
            "observation_semantics": contract["observation_semantics"],
        }
        if self.cfg.protocol == "lh_mtlc":
            value["sequence_count"] = len(self._sequences)
        else:
            value["mtlc_trial_count"] = len(self._mtlc_trials)
        return value

    def _check_task(self, task_id: int) -> None:
        if isinstance(task_id, bool) or not isinstance(task_id, (int, np.integer)):
            raise IndexError(f"CALVIN task_id must be integer, got {task_id!r}")
        if not 0 <= int(task_id) < len(self._sequences):
            raise IndexError(f"CALVIN task_id out of range: {task_id}")

    def _ensure_env(self) -> Any:
        if self._env is None:
            if self._env_factory is not None:
                self._env = self._env_factory(self.validation_path)
            else:
                _apply_numpy_compat()
                from calvin_env.envs.play_table_env import get_env

                self._env = get_env(
                    self.validation_path,
                    obs_space=self.CAMERA_OBS_SPACE,
                    show_gui=False,
                )
        return self._env

    def reset_episode(self, task_id: int, init_state_id: int, run_index: int, episode_seed: int) -> dict[str, Any]:
        self._check_task(task_id)
        if init_state_id != 0:
            raise IndexError("CALVIN adapter exposes exactly init_state_id=0 per Forge task")
        env = self._ensure_env()
        initial_state, sequence = self._sequences[task_id]

        if self.cfg.protocol == "mtlc":
            task, dataset_id = self._mtlc_trials[task_id]
            episode_file = self._mtlc_frame_file(dataset_id)
            with np.load(episode_file, allow_pickle=False) as data:
                if "robot_obs" not in data or "scene_obs" not in data:
                    raise KeyError(f"MTLC validation frame lacks robot_obs/scene_obs: {episode_file}")
                robot_obs = np.asarray(data["robot_obs"]).copy()
                scene_obs = np.asarray(data["scene_obs"]).copy()
            sequence = [task]
        else:
            from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

        self._obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        start_info = env.get_info()
        self._task_id = int(task_id)
        self._episode_metrics = {"sequence_length": len(sequence), "chain_length": 0, "subtasks": list(sequence)}

        if self.cfg.protocol == "lh_mtlc":
            self._protocol = LhMtlcProtocol(tuple(sequence), self._language, max_actions=LH_MAX_ACTIONS)
            self._protocol.begin(start_info)
            self._lh_actions = 0
        else:
            self._protocol = None
            if self._mtlc_protocol is None:
                raise RuntimeError("MTLC protocol is not initialized")
            self._mtlc_protocol.begin(str(self._mtlc_trials[task_id][0]), start_info)
            self._episode_metrics["steps"] = 0
            self._episode_metrics["dataset_id"] = self._mtlc_trials[task_id][1]
            self._mtlc_actions = 0
        return self._obs

    def step(self, action: np.ndarray) -> StepOutcome:
        env = self._ensure_env()
        if self._obs is None or self._task_id is None:
            raise RuntimeError("reset_episode must be called before step")
        encoded = self.codec.encode(action)
        obs, reward, _env_done, _info = env.step(encoded)
        self._obs = obs
        if self.cfg.protocol == "lh_mtlc":
            return self._lh_step(obs, reward, env)
        return self._mtlc_step(obs, reward, env)

    def _lh_step(self, obs: dict[str, Any], reward: float, env: Any) -> StepOutcome:
        if self._protocol is None:
            raise RuntimeError("LH-MTLC protocol is not initialized")
        # Official evaluator ignores env reward/done and checks TaskOracle
        # after every executed action, including action 360.
        result = self._protocol.observe(env.get_info(), self._oracle)
        self._episode_metrics.update(result)
        # Publish the current sub-task deadline through the shared runner's
        # ``max_steps``.  When the official 360-action budget is exhausted the
        # deadline lands exactly on the current cumulative step, so the frozen
        # runner records ``timed_out`` instead of the default ``partial`` label
        # (manual §3.4 rule 3 / §8 "three states distinguishable").  Scored
        # quantities (``chain_length``, ``num_steps``, ``success``) are unchanged.
        self._lh_actions += 1
        self.cfg.max_steps = self._lh_actions + (
            self._protocol.max_actions - self._protocol.actions
        )
        timed_out = bool(result.get("timed_out_subtask"))
        return StepOutcome(
            obs=obs,
            reward=float(reward),
            done=bool(result["done"]) and not timed_out,
            success=bool(result["success"]),
            metrics=dict(self._episode_metrics),
            next_instruction=result.get("next_instruction"),
        )

    def _mtlc_step(self, obs: dict[str, Any], reward: float, env: Any) -> StepOutcome:
        if self._mtlc_protocol is None:
            raise RuntimeError("MTLC protocol is not initialized")
        result = self._mtlc_protocol.observe(env.get_info(), self._oracle)
        self._episode_metrics.update(
            {
                "steps": result["steps"],
                "chain_length": result["chain_length"],
                "subtasks": result["subtasks"],
            }
        )
        if result.get("failed_subtask"):
            self._episode_metrics["failed_subtask"] = result["failed_subtask"]
        # Same deadline publication as ``_lh_step``: the official 240-action
        # budget is reported as ``timed_out`` at the exact final step.
        self._mtlc_actions += 1
        self.cfg.max_steps = self._mtlc_actions + (
            self._mtlc_protocol.max_actions - result["steps"]
        )
        timed_out = bool(result["done"] and not result["success"])
        return StepOutcome(
            obs=obs,
            reward=float(reward),
            done=bool(result["done"]) and not timed_out,
            success=bool(result["success"]),
            metrics=dict(self._episode_metrics),
        )

    def to_state(self, obs: dict[str, Any]) -> np.ndarray:
        if "robot_obs" not in obs:
            raise KeyError("CALVIN observation missing robot_obs")
        state = np.asarray(obs["robot_obs"], dtype=np.float64).reshape(-1)
        if state.size != len(self._state_names):
            raise ValueError(f"robot_obs has {state.size} values; expected {len(self._state_names)}")
        return state

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        rgb = obs.get("rgb_obs")
        if not isinstance(rgb, Mapping):
            raise KeyError("CALVIN observation missing rgb_obs mapping")
        out: dict[str, np.ndarray] = {}
        for source, target in (("rgb_static", "image/main"), ("rgb_gripper", "image/second")):
            if source not in rgb:
                raise KeyError(f"CALVIN observation missing {source}")
            frame = np.asarray(rgb[source])
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{source} must be HWC RGB, got {frame.shape} {frame.dtype}")
            if frame.dtype != np.uint8:
                if not np.issubdtype(frame.dtype, np.integer):
                    raise ValueError(f"{source} must contain integer RGB pixels, got {frame.dtype}")
                if frame.size and (int(frame.min()) < 0 or int(frame.max()) > 255):
                    raise ValueError(f"{source} integer RGB values are outside uint8 range")
                # Gate-A on the actual CALVIN EGL renderer measured int64
                # values in 0..255.  Forge Image requires an 8-bit RGB carrier;
                # this cast changes representation only, never pixel values.
                frame = frame.astype(np.uint8, copy=False)
            out[target] = np.ascontiguousarray(frame)
        return out

    def aggregate(self, episode_metrics: list[dict[str, Any]]) -> tuple[dict, dict]:
        total = len(episode_metrics)
        if self.cfg.protocol == "mtlc":
            successes = sum(int(m.get("chain_length", 0)) == 1 for m in episode_metrics)
            planned = len(self.cfg.episode_plan()) if hasattr(self.cfg, "episode_plan") else total
            success_rate = successes / planned if planned else 0.0
            per_task: dict[str, dict[str, int]] = {}
            for metrics in episode_metrics:
                subtasks = list(metrics.get("subtasks", []))
                if len(subtasks) != 1:
                    raise ValueError("MTLC episode metrics must contain exactly one task")
                row = per_task.setdefault(str(subtasks[0]), {"success": 0, "total": 0})
                row["total"] += 1
                row["success"] += int(metrics.get("chain_length", 0)) == 1
            return (
                {"name": "success_rate", "value": success_rate, "max": 1.0},
                {
                    "success_rate": success_rate,
                    "completed_episode_count": total,
                    "planned_episode_count": planned,
                    "per_task": per_task,
                },
            )

        # Official CALVIN LH-MTLC aggregation (count_success/print_and_save).
        lengths = [int(m.get("chain_length", 0)) for m in episode_metrics]
        if any(length < 0 or length > 5 for length in lengths):
            raise ValueError("LH-MTLC chain_length must be in [0,5]")
        avg = sum(lengths) / total if total else 0.0
        extra: dict[str, Any] = {"avg_seq_len": avg, "sequence_count": total}
        for n in range(1, 6):
            extra[f"SR{n}"] = sum(length >= n for length in lengths) / total if total else 0.0

        per_task: dict[str, dict[str, int]] = {}
        for metrics, chain in zip(episode_metrics, lengths):
            subtasks = list(metrics.get("subtasks", []))
            if len(subtasks) != 5:
                raise ValueError("LH-MTLC episode metrics must contain exactly five subtasks")
            for position, task in enumerate(subtasks):
                if position > chain:
                    break  # tasks after the first failure were never attempted
                row = per_task.setdefault(str(task), {"success": 0, "total": 0})
                row["total"] += 1
                if position < chain:
                    row["success"] += 1
                else:
                    break  # position == chain is the first failed task
        extra["per_task"] = per_task

        planned = len(self.cfg.episode_plan()) if hasattr(self.cfg, "episode_plan") else total
        full_chain_successes = sum(length == 5 for length in lengths)
        extra["success_rate"] = full_chain_successes / planned if planned else 0.0
        extra["planned_sequence_count"] = planned
        return ({"name": "avg_sequence_length", "value": avg, "max": 5.0}, extra)

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 关闭失败不该影响批次收尾与落盘
                    pass
        self._env = None
        self._obs = None
        self._protocol = None
        self._mtlc_protocol = None
