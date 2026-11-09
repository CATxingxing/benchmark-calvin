"""Official CALVIN MCIL inference, isolated from the benchmark process.

Loading and preprocessing follow CALVIN fa03f01 (MIT), and were cross-checked
against the previously delivered validation binary. This source is maintained
as part of the benchmark's MCIL evaluation profile; it is not a general robot
policy. See POLICY_PROVENANCE.md for the original baseline and intentional fixes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def _find_calvin_root() -> Path:
    root = os.environ.get("CALVIN_ROOT")
    if not root:
        raise RuntimeError("CALVIN_ROOT must point at the CALVIN checkout")
    return Path(root).expanduser().resolve()


def bootstrap_calvin_paths() -> Path:
    base = _find_calvin_root()
    for path in (base / "calvin_env", base / "calvin_models", base / "calvin_env" / "tacto"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return base


def calvin_conf_root(calvin_root: Path) -> Path:
    import calvin_agent
    return Path(calvin_agent.__file__).resolve().parents[1] / "conf"


def build_val_dataset(cfg: Any, dataset_path: str) -> Any:
    import hydra
    from hydra import compose, initialize_config_dir

    conf_root = calvin_conf_root(_find_calvin_root())
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    with initialize_config_dir(version_base=None, config_dir=str(conf_root / "datamodule" / "datasets")):
        datasets_cfg = compose(config_name="vision_lang.yaml", overrides=[f"lang_dataset.lang_folder={lang_folder}"])
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    return data_module.val_dataloader().dataset.datasets["lang"]


def load_mcil(cfg: Any, checkpoint: str, dataset: Any, device: torch.device) -> Any:
    from pydoc import locate
    model_cls = locate(cfg.model._target_)
    if model_cls is None:
        raise RuntimeError(f"Cannot import checkpoint model class {cfg.model._target_}")
    model = model_cls.load_from_checkpoint(str(checkpoint))
    model.load_lang_embeddings(Path(dataset.abs_datasets_dir) / dataset.lang_folder / "embeddings.npy")
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.to(device)
    model.eval()
    return model


def build_official_obs(dataset: Any, raw_robot_obs: np.ndarray, rgb_static: np.ndarray,
                       rgb_gripper: np.ndarray | None, device: torch.device) -> dict[str, Any]:
    from calvin_agent.datasets.utils.episode_utils import process_depth, process_rgb, process_state
    raw = {
        "robot_obs": np.asarray(raw_robot_obs, dtype=np.float32),
        "rgb_obs": {"rgb_static": np.asarray(rgb_static, dtype=np.uint8)},
        "depth_obs": {},
    }
    if rgb_gripper is not None:
        raw["rgb_obs"]["rgb_gripper"] = np.asarray(rgb_gripper, dtype=np.uint8)
    state_obs = process_state(raw, dataset.observation_space, dataset.transforms, dataset.proprio_state)
    rgb_obs = process_rgb(raw["rgb_obs"], dataset.observation_space, dataset.transforms)
    depth_obs = process_depth(raw["depth_obs"], dataset.observation_space, dataset.transforms)
    state_obs["robot_obs"] = state_obs["robot_obs"].to(device).unsqueeze(0)
    rgb_obs.update({"rgb_obs": {k: v.to(device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
    depth_obs.update({"depth_obs": {k: v.to(device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})
    return {**rgb_obs, **state_obs, **depth_obs, "robot_obs_raw": torch.from_numpy(raw["robot_obs"]).to(device)}


class MCILRuntime:
    """Stateful MCIL inference: exactly one model.step per complete observation."""
    def __init__(self, train_folder: str, checkpoint: str, dataset_path: str, device: str):
        bootstrap_calvin_paths()
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
        self.dataset = build_val_dataset(self.cfg, dataset_path)
        self.model = load_mcil(self.cfg, checkpoint, self.dataset, self.device)
        self.instruction: str | None = None
        self.action_dim = int(getattr(self.model.action_decoder, "out_features", 7))

    def reset(self) -> None:
        self.model.reset()

    def set_instruction(self, instruction: str) -> None:
        if instruction not in self.model.lang_embeddings:
            raise KeyError(f"instruction {instruction!r} is absent from the checkpoint language embeddings; refusing to run with a silently wrong goal")
        self.instruction = instruction

    @torch.no_grad()
    def step(self, raw_robot_obs: np.ndarray, rgb_static: np.ndarray, rgb_gripper: np.ndarray | None) -> np.ndarray:
        if self.instruction is None:
            raise RuntimeError("set_instruction must be called before step")
        obs = build_official_obs(self.dataset, raw_robot_obs, rgb_static, rgb_gripper, self.device)
        action = self.model.step(obs, self.instruction)
        return np.asarray(action.detach().cpu().reshape(-1), dtype=np.float32)
