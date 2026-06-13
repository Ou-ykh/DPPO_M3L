from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from utils.omnireset import OmniResetSampler


STATE_SITE_NAMES = ("center_point", "finger_left", "finger_right")
STATE_BODY_NAMES = ("object", "walls")
STATE_EXTRA_DIM = 21


def normalize_asset_paths_for_mujoco(xml_path: Path, repo_root: Path) -> tuple[str, int]:
    """Return XML text with old Windows asset paths rewritten under repo_root.

    Absolute POSIX paths are preserved. This lets server-side XMLs keep paths
    like /home/users/... while still allowing old G:\\DPPO\\... files to be
    normalized when needed.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = 0
    xml_dir = xml_path.parent

    for asset in root.findall(".//asset/*[@file]"):
        raw = asset.get("file")
        if not raw:
            continue

        normalized = raw.replace("\\", "/")
        lowered = normalized.lower()
        marker = "dppo/"
        marker_index = lowered.find(marker)

        if marker_index >= 0:
            rel = normalized[marker_index + len(marker) :]
            rewritten = repo_root / Path(*rel.split("/"))
        elif ":" in normalized[:4]:
            rewritten = Path(raw)
        elif normalized.startswith("/"):
            continue
        elif Path(normalized).is_absolute():
            rewritten = Path(normalized)
        else:
            rewritten = xml_dir / Path(*normalized.split("/"))

        rewritten_str = str(rewritten)
        if rewritten_str != raw:
            asset.set("file", rewritten_str)
            changed += 1

    return ET.tostring(root, encoding="unicode"), changed


def load_mj_model(
    xml_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str] | None = None,
    normalize_assets: bool = True,
):
    import mujoco

    path = Path(xml_path).expanduser().resolve()
    if not normalize_assets:
        return mujoco.MjModel.from_xml_path(str(path)), 0

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    xml_string, changed = normalize_asset_paths_for_mujoco(path, root)
    return mujoco.MjModel.from_xml_string(xml_string), changed


def mjwarp_state_dim(mj_model) -> int:
    return int(mj_model.nq + mj_model.nv + mj_model.nu + STATE_EXTRA_DIM)


def make_state_policy_config(config: Any, state_dim: int, action_dim: int) -> Dict[str, Any]:
    return {
        "policy_kind": "dppo",
        "state_dim": int(state_dim),
        "dim_embeddings": int(config.dim_embedding),
        "action_dim": int(action_dim),
        "action_horizon": int(config.action_horizon),
        "state_encoder_hidden_dim": int(config.state_encoder_hidden_dim),
        "state_encoder_depth": int(config.state_encoder_depth),
        "actor_hidden_dim": int(config.diffusion_hidden_dim),
        "critic_hidden_dim": int(config.diffusion_hidden_dim),
        "denoising_steps": int(config.denoising_steps),
        "diffusion_time_dim": int(config.diffusion_time_dim),
        "diffusion_denoiser_type": str(config.diffusion_denoiser_type),
        "diffusion_transformer_depth": int(config.diffusion_transformer_depth),
        "diffusion_transformer_heads": int(config.diffusion_transformer_heads),
        "diffusion_transformer_dropout": float(config.diffusion_transformer_dropout),
        "diffusion_cnn_depth": int(config.diffusion_cnn_depth),
        "diffusion_cnn_kernel_size": int(config.diffusion_cnn_kernel_size),
        "diffusion_cnn_dropout": float(config.diffusion_cnn_dropout),
        "beta_start": float(config.diffusion_beta_start),
        "beta_end": float(config.diffusion_beta_end),
        "min_sampling_std": float(config.min_sampling_std),
        "min_logprob_std": float(config.min_logprob_std),
    }


def make_mlp_policy_config(config: Any, state_dim: int, action_dim: int) -> Dict[str, Any]:
    return {
        "policy_kind": "mlp",
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "action_horizon": int(config.action_horizon),
        "hidden_dim": int(config.mlp_hidden_dim),
        "depth": int(config.mlp_depth),
        "log_std_init": float(config.mlp_log_std_init),
    }


def write_policy_config(path: str | os.PathLike[str], config: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)


def read_policy_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _plain_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        array = value.numpy()
    elif hasattr(value, "to_numpy"):
        array = value.to_numpy()
    else:
        array = np.asarray(value)

    array = np.asarray(array)
    if array.dtype.fields:
        names = list(array.dtype.names or [])
        array = np.stack([array[name] for name in names], axis=-1)
    elif array.dtype == object:
        array = np.asarray(array.tolist())

    return array.astype(np.float32, copy=False)


def _copy_numpy_to_warp(wp, dst, value: np.ndarray) -> None:
    src = wp.array(np.ascontiguousarray(value), dtype=wp.float32, device=dst.device)
    wp.copy(dst, src)


def _name_id(mujoco, mj_model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(mj_model, obj_type, name)
    if idx < 0:
        raise ValueError(f"missing MJCF {obj_type} named '{name}'")
    return int(idx)


class MJWarpStateRollout:
    """State-only rollout collector backed by native MuJoCo Warp."""

    def __init__(
        self,
        xml_path: str | os.PathLike[str],
        n_envs: int,
        n_steps: int,
        seed: int = 0,
        nconmax: Optional[int] = 64,
        naconmax: Optional[int] = None,
        njmax: int = 512,
        gamma: float = 0.99,
        action_exec_index: int = 0,
        action_exec_steps: int = 1,
        sim_substeps: int = 25,
        repo_root: str | os.PathLike[str] | None = None,
        normalize_assets: bool = True,
        reward_success_tolerance: float = 2.5e-3,
        reward_success_bonus: float = 1000.0,
        reward_distance_coef: float = 5.0,
        reward_time_penalty: float = 0.5,
        reward_gripper_coef: float = 0.2,
        reward_smoothness_coef: float = 0.0,
        reward_truncation_penalty: float = 0.0,
        reset_types: str | Iterable[str] | None = None,
        reset_stabilize_steps: int = 32,
        reset_max_sample_attempts: int = 64,
        reset_bank_size_per_type: int = 64,
        reset_cache_dir: str | os.PathLike[str] | None = None,
        reset_force_rebuild: bool = False,
        reset_grasp_close_steps: int = 20,
        reset_goal_offset_bank_size: int = 64,
        reset_goal_offset_perturb_steps: int = 12,
        arm_action_scale: float = 0.1,
        gripper_action_scale: float = 1.0,
    ):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if action_exec_steps < 1:
            raise ValueError(f"action_exec_steps must be >= 1, got {action_exec_steps}")
        if sim_substeps < 1:
            raise ValueError(f"sim_substeps must be >= 1, got {sim_substeps}")

        self.mujoco = mujoco
        self.mjw = mjw
        self.wp = wp
        self.n_envs = int(n_envs)
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.action_exec_index = int(action_exec_index)
        self.action_exec_steps = int(action_exec_steps)
        self.sim_substeps = int(sim_substeps)
        self.reward_success_tolerance = float(reward_success_tolerance)
        self.reward_success_bonus = float(reward_success_bonus)
        self.reward_distance_coef = float(reward_distance_coef)
        self.reward_time_penalty = float(reward_time_penalty)
        self.reward_gripper_coef = float(reward_gripper_coef)
        self.reward_smoothness_coef = float(reward_smoothness_coef)
        self.reward_truncation_penalty = float(reward_truncation_penalty)
        self.reset_bank_size_per_type = int(reset_bank_size_per_type)
        self.reset_cache_dir = reset_cache_dir
        self.reset_force_rebuild = bool(reset_force_rebuild)
        self.arm_action_scale = float(arm_action_scale)
        self.gripper_action_scale = float(gripper_action_scale)

        self.mj_model, self.rewritten_assets = load_mj_model(
            xml_path,
            repo_root=repo_root,
            normalize_assets=normalize_assets,
        )
        self.state_dim = mjwarp_state_dim(self.mj_model)
        self.action_dim = int(self.mj_model.nu)
        self.ctrl_low, self.ctrl_high, self.ctrl_scale = self._control_ranges()

        self.site_ids = {
            name: _name_id(mujoco, self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in STATE_SITE_NAMES
        }
        self.body_ids = {
            name: _name_id(mujoco, self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in STATE_BODY_NAMES
        }

        make_kwargs = {"nworld": self.n_envs, "njmax": int(njmax)}
        if naconmax is not None:
            make_kwargs["naconmax"] = int(naconmax)
        elif nconmax is not None:
            make_kwargs["nconmax"] = int(nconmax)

        self.reset_sampler = OmniResetSampler(
            self.mj_model,
            seed=self.seed,
            reset_types=reset_types,
            stabilize_steps=reset_stabilize_steps,
            max_sample_attempts=reset_max_sample_attempts,
            grasp_close_steps=reset_grasp_close_steps,
            goal_offset_bank_size=reset_goal_offset_bank_size,
            goal_offset_perturb_steps=reset_goal_offset_perturb_steps,
        )
        self.reset_bank_cache_path = self._make_reset_cache_path(xml_path)
        if self.reset_bank_cache_path is not None:
            self.reset_bank_cache_path = self.reset_sampler._resolve_cache_path(
                self.reset_bank_cache_path,
                self.reset_bank_size_per_type,
            )
        self.reset_bank = self.reset_sampler.build_bank(
            self.reset_bank_size_per_type,
            cache_path=self.reset_bank_cache_path,
            force_rebuild=self.reset_force_rebuild,
        )
        self.reset_bank_sizes = {
            reset_type: int(entry["qpos"].shape[0]) for reset_type, entry in self.reset_bank.items()
        }
        self.invalid_reset_events = 0
        self.invalid_reset_worlds = 0
        self.obs_clip = 1e3

        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.put_data(self.mj_model, self._make_initial_mj_data(), **make_kwargs)
        self._last_ctrl = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)
        self._last_reset_types = [""] * self.n_envs
        self.reset_worlds(np.arange(self.n_envs, dtype=np.int32))
        self.wp.synchronize()
        if hasattr(self.mjw, "forward"):
            self.mjw.forward(self.model, self.data)
            self.wp.synchronize()

    def _make_reset_cache_path(self, xml_path: str | os.PathLike[str]) -> Path | None:
        if self.reset_cache_dir is None:
            return None
        cache_root = Path(self.reset_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        stem = Path(xml_path).stem
        return cache_root / stem

    def _make_initial_mj_data(self):
        data = self.mujoco.MjData(self.mj_model)
        if getattr(self.mj_model, "nkey", 0) > 0:
            data.qpos[:] = self.mj_model.key_qpos[0]
            if self.mj_model.nv:
                data.qvel[:] = 0.0
            if self.mj_model.nu:
                data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.mj_model, data)
        return data

    def reset_worlds(self, world_ids) -> None:
        world_ids = np.asarray(world_ids, dtype=np.int32).reshape(-1)
        if world_ids.size == 0:
            return

        reset_batch = self.reset_sampler.sample_batch(world_ids.size, bank=self.reset_bank)
        qpos = _plain_numpy(self.data.qpos)
        qvel = _plain_numpy(self.data.qvel)
        ctrl = _plain_numpy(self.data.ctrl)

        qpos[world_ids] = np.asarray(reset_batch["qpos"], dtype=np.float32)
        qvel[world_ids] = np.asarray(reset_batch["qvel"], dtype=np.float32)
        ctrl[world_ids] = np.asarray(reset_batch["ctrl"], dtype=np.float32)
        self._last_ctrl[world_ids] = ctrl[world_ids]
        for target_index, reset_type in zip(world_ids.tolist(), reset_batch["reset_types"]):
            self._last_reset_types[target_index] = str(reset_type)

        _copy_numpy_to_warp(self.wp, self.data.qpos, qpos)
        _copy_numpy_to_warp(self.wp, self.data.qvel, qvel)
        _copy_numpy_to_warp(self.wp, self.data.ctrl, ctrl)
        if hasattr(self.mjw, "forward"):
            self.mjw.forward(self.model, self.data)
        self.wp.synchronize()

    def _control_ranges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mj_model.nu == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty, empty

        ranges = np.asarray(self.mj_model.actuator_ctrlrange, dtype=np.float32)
        lows = ranges[:, 0]
        highs = ranges[:, 1]
        scale = np.maximum(np.abs(lows), np.abs(highs))
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype(np.float32)
        scale_factors = np.ones_like(scale, dtype=np.float32)
        if scale_factors.size > 1:
            scale_factors[:-1] *= self.arm_action_scale
            scale_factors[-1] *= self.gripper_action_scale
        elif scale_factors.size == 1:
            scale_factors[0] *= self.gripper_action_scale
        scale = scale * scale_factors
        return lows, highs, scale

    def _actions_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_envs, self.action_dim):
            raise ValueError(
                f"action shape must be {(self.n_envs, self.action_dim)}, got {action.shape}"
            )
        ctrl = np.clip(action, -1.0, 1.0) * self.ctrl_scale[None, :]
        ctrl = np.clip(ctrl, self.ctrl_low[None, :], self.ctrl_high[None, :])
        return ctrl.astype(np.float32, copy=False)

    def _smoothness_penalty(self, previous_ctrl: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        if self.reward_smoothness_coef <= 0.0 or self.action_dim == 0:
            return np.zeros((self.n_envs,), dtype=np.float32)
        denom = np.maximum(np.abs(self.ctrl_scale), 1e-8).astype(np.float32)
        normalized_delta = (ctrl - previous_ctrl) / denom[None, :]
        normalized_delta = np.nan_to_num(
            normalized_delta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        penalty = self.reward_smoothness_coef * np.mean(
            normalized_delta * normalized_delta,
            axis=-1,
        )
        return penalty.astype(np.float32, copy=False)

    def _apply_truncation_penalty(self, reward: np.ndarray, world_ids: np.ndarray) -> np.ndarray:
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        if world_ids.size:
            if self.reward_truncation_penalty > 0.0:
                reward[world_ids] = -self.reward_truncation_penalty
            else:
                reward[world_ids] = 0.0
        return reward.astype(np.float32, copy=False)

    def _positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        site_xpos = _plain_numpy(self.data.site_xpos)
        xpos = _plain_numpy(self.data.xpos)
        center = site_xpos[:, self.site_ids["center_point"]]
        finger_left = site_xpos[:, self.site_ids["finger_left"]]
        finger_right = site_xpos[:, self.site_ids["finger_right"]]
        obj_pos = xpos[:, self.body_ids["object"]]
        wall_pos = xpos[:, self.body_ids["walls"]]
        return center, finger_left, finger_right, obj_pos, wall_pos

    def _invalid_world_ids_from_arrays(self, *arrays: np.ndarray) -> np.ndarray:
        invalid = np.zeros(self.n_envs, dtype=bool)
        for array in arrays:
            reshaped = np.asarray(array).reshape(self.n_envs, -1)
            invalid |= ~np.isfinite(reshaped).all(axis=1)
        return np.flatnonzero(invalid)

    def _maybe_log_invalid_worlds(self, reason: str, world_ids: np.ndarray) -> None:
        if world_ids.size == 0:
            return
        self.invalid_reset_events += 1
        self.invalid_reset_worlds += int(world_ids.size)
        if self.invalid_reset_events <= 5 or self.invalid_reset_events % 20 == 0:
            print(
                "[MJWarp rollout] resetting "
                f"{world_ids.size} non-finite worlds after {reason} "
                f"(events={self.invalid_reset_events}, worlds={self.invalid_reset_worlds})",
                flush=True,
            )

    def observe(self) -> Dict[str, np.ndarray]:
        qpos = _plain_numpy(self.data.qpos)
        qvel = _plain_numpy(self.data.qvel)
        center, finger_left, finger_right, obj_pos, wall_pos = self._positions()
        gripper_center = 0.5 * (finger_left + finger_right)
        state = np.concatenate(
            (
                qpos,
                qvel,
                self._last_ctrl,
                center,
                finger_left,
                finger_right,
                obj_pos,
                wall_pos,
                obj_pos - wall_pos,
                gripper_center - obj_pos,
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        invalid_world_ids = self._invalid_world_ids_from_arrays(state)
        if invalid_world_ids.size:
            self._maybe_log_invalid_worlds("observe", invalid_world_ids)
            self.reset_worlds(invalid_world_ids)
            qpos = _plain_numpy(self.data.qpos)
            qvel = _plain_numpy(self.data.qvel)
            center, finger_left, finger_right, obj_pos, wall_pos = self._positions()
            gripper_center = 0.5 * (finger_left + finger_right)
            state = np.concatenate(
                (
                    qpos,
                    qvel,
                    self._last_ctrl,
                    center,
                    finger_left,
                    finger_right,
                    obj_pos,
                    wall_pos,
                    obj_pos - wall_pos,
                    gripper_center - obj_pos,
                ),
                axis=-1,
            ).astype(np.float32, copy=False)
        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=self.obs_clip,
            neginf=-self.obs_clip,
        )
        state = np.clip(state, -self.obs_clip, self.obs_clip)
        if state.shape[-1] != self.state_dim:
            raise RuntimeError(f"state_dim mismatch: expected {self.state_dim}, got {state.shape[-1]}")
        return {"state": state}

    def _reward_done(self) -> tuple[np.ndarray, np.ndarray]:
        center, finger_left, finger_right, obj_pos, wall_pos = self._positions()
        gripper_center = 0.5 * (finger_left + finger_right)
        site_obj = np.linalg.norm(obj_pos - center, axis=-1)
        obj_wall = np.linalg.norm(obj_pos - wall_pos, axis=-1)
        gripper_obj = np.linalg.norm(obj_pos - gripper_center, axis=-1)
        distance_reward = -self.reward_distance_coef * (
            np.log(site_obj * site_obj + 1.0)
            + np.log(obj_wall * obj_wall + 1.0)
            + 0.2 * np.log(gripper_obj * gripper_obj + 1.0)
        )
        gripper_reward = np.zeros_like(distance_reward, dtype=np.float32)
        if self.action_dim:
            gripper_reward = self.reward_gripper_coef * np.clip(
                (self._last_ctrl[:, -1] + self.ctrl_scale[-1]) / (2.0 * self.ctrl_scale[-1] + 1e-8),
                0.0,
                1.0,
            )
        done = obj_wall < self.reward_success_tolerance
        reward = (
            distance_reward
            + gripper_reward
            - self.reward_time_penalty
            + done.astype(np.float32) * self.reward_success_bonus
        )
        invalid_world_ids = self._invalid_world_ids_from_arrays(
            center,
            finger_left,
            finger_right,
            obj_pos,
            wall_pos,
            reward[:, None],
        )
        if invalid_world_ids.size:
            reward = self._apply_truncation_penalty(reward, invalid_world_ids)
            done = done.astype(np.float32)
            done[invalid_world_ids] = 1.0
            return reward.astype(np.float32), done.astype(np.float32)
        return reward.astype(np.float32), done.astype(np.float32)

    def step_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        previous_ctrl = self._last_ctrl.copy()
        ctrl = self._actions_to_ctrl(action)
        smoothness_penalty = self._smoothness_penalty(previous_ctrl, ctrl)
        self._last_ctrl = ctrl.copy()
        _copy_numpy_to_warp(self.wp, self.data.ctrl, ctrl)
        for _ in range(self.sim_substeps):
            self.mjw.step(self.model, self.data)
        self.wp.synchronize()
        reward, done = self._reward_done()
        if self.reward_smoothness_coef > 0.0:
            reward = reward - smoothness_penalty
        qpos = _plain_numpy(self.data.qpos)
        qvel = _plain_numpy(self.data.qvel)
        ctrl = _plain_numpy(self.data.ctrl)
        invalid_world_ids = self._invalid_world_ids_from_arrays(qpos, qvel, ctrl)
        if invalid_world_ids.size:
            self._maybe_log_invalid_worlds("step", invalid_world_ids)
            reward = self._apply_truncation_penalty(reward, invalid_world_ids)
            done[invalid_world_ids] = 1.0
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        done = np.nan_to_num(done, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        return reward, done

    def collect(self, policy, device: str | Any = "cuda") -> Dict[str, Any]:
        if hasattr(policy, "denoising_steps"):
            return self._collect_diffusion(policy, device=device)
        return self._collect_mlp(policy, device=device)

    def _collect_diffusion(self, policy, device: str | Any = "cuda") -> Dict[str, Any]:
        import torch

        if self.action_exec_index < 0 or self.action_exec_index >= policy.action_horizon:
            raise ValueError(
                f"action_exec_index must be in [0, {policy.action_horizon - 1}], "
                f"got {self.action_exec_index}"
            )
        if self.action_exec_index + self.action_exec_steps > policy.action_horizon:
            raise ValueError(
                "action_exec_index + action_exec_steps must be <= action_horizon "
                f"({self.action_exec_index} + {self.action_exec_steps} > {policy.action_horizon})"
            )

        obs_buffer = {
            "state": np.zeros((self.n_steps, self.n_envs, self.state_dim), dtype=np.float32)
        }
        chains_buffer = np.zeros(
            (
                self.n_steps,
                self.n_envs,
                policy.denoising_steps + 1,
                policy.action_chunk_dim,
            ),
            dtype=np.float32,
        )
        log_probs_buffer = np.zeros(
            (self.n_steps, self.n_envs, policy.denoising_steps), dtype=np.float32
        )
        values_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        rewards_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        dones_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        discounts_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        policy.eval()
        current_obs = self.observe()
        for step in range(self.n_steps):
            obs_buffer["state"][step] = current_obs["state"]
            obs_torch = {
                "state": torch.as_tensor(current_obs["state"], dtype=torch.float32, device=device)
            }
            with torch.no_grad():
                actions, chains, log_probs, values = policy.act(obs_torch, deterministic=False)

            action_chunks = actions.detach().cpu().numpy()
            chains_buffer[step] = chains.detach().cpu().numpy()
            log_probs_buffer[step] = log_probs.detach().cpu().numpy()
            values_buffer[step] = values.detach().cpu().numpy()

            active = np.ones(self.n_envs, dtype=bool)
            discounts = np.ones(self.n_envs, dtype=np.float32)
            for exec_offset in range(self.action_exec_steps):
                action_index = self.action_exec_index + exec_offset
                action = action_chunks[:, action_index].copy()
                action[~active] = 0.0
                reward, done = self.step_action(action)
                reward = np.where(active, reward, 0.0).astype(np.float32)
                done_bool = (done > 0.0) & active

                rewards_buffer[step] += (self.gamma ** exec_offset) * reward
                dones_buffer[step] = np.maximum(dones_buffer[step], done_bool.astype(np.float32))
                discounts[active] = self.gamma ** (exec_offset + 1)
                discounts[done_bool] = 0.0
                active[done_bool] = False
                if not active.any():
                    break

            discounts_buffer[step] = discounts
            done_world_ids = np.flatnonzero(dones_buffer[step] > 0.0)
            if done_world_ids.size:
                self.reset_worlds(done_world_ids)
            current_obs = self.observe()

        obs_torch = {
            "state": torch.as_tensor(current_obs["state"], dtype=torch.float32, device=device)
        }
        with torch.no_grad():
            next_values = policy.value(obs_torch).detach().cpu().numpy().astype(np.float32)

        return {
            "obs": obs_buffer,
            "chains": chains_buffer,
            "old_log_probs": log_probs_buffer,
            "old_values": values_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "discounts": discounts_buffer,
            "next_values": next_values,
        }

    def _collect_mlp(self, policy, device: str | Any = "cuda") -> Dict[str, Any]:
        import torch

        if self.action_exec_index < 0 or self.action_exec_index >= policy.action_horizon:
            raise ValueError(
                f"action_exec_index must be in [0, {policy.action_horizon - 1}], "
                f"got {self.action_exec_index}"
            )
        if self.action_exec_index + self.action_exec_steps > policy.action_horizon:
            raise ValueError(
                "action_exec_index + action_exec_steps must be <= action_horizon "
                f"({self.action_exec_index} + {self.action_exec_steps} > {policy.action_horizon})"
            )

        obs_buffer = {
            "state": np.zeros((self.n_steps, self.n_envs, self.state_dim), dtype=np.float32)
        }
        actions_buffer = np.zeros(
            (
                self.n_steps,
                self.n_envs,
                policy.action_horizon,
                policy.action_dim,
            ),
            dtype=np.float32,
        )
        log_probs_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        values_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        rewards_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        dones_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        discounts_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        policy.eval()
        current_obs = self.observe()
        for step in range(self.n_steps):
            obs_buffer["state"][step] = current_obs["state"]
            obs_torch = {
                "state": torch.as_tensor(current_obs["state"], dtype=torch.float32, device=device)
            }
            with torch.no_grad():
                actions, log_probs, values = policy.act(obs_torch, deterministic=False)

            action_chunks = actions.detach().cpu().numpy()
            actions_buffer[step] = action_chunks
            log_probs_buffer[step] = log_probs.detach().cpu().numpy()
            values_buffer[step] = values.detach().cpu().numpy()

            active = np.ones(self.n_envs, dtype=bool)
            discounts = np.ones(self.n_envs, dtype=np.float32)
            for exec_offset in range(self.action_exec_steps):
                action_index = self.action_exec_index + exec_offset
                action = action_chunks[:, action_index].copy()
                action[~active] = 0.0
                reward, done = self.step_action(action)
                reward = np.where(active, reward, 0.0).astype(np.float32)
                done_bool = (done > 0.0) & active

                rewards_buffer[step] += (self.gamma ** exec_offset) * reward
                dones_buffer[step] = np.maximum(dones_buffer[step], done_bool.astype(np.float32))
                discounts[active] = self.gamma ** (exec_offset + 1)
                discounts[done_bool] = 0.0
                active[done_bool] = False
                if not active.any():
                    break

            discounts_buffer[step] = discounts
            done_world_ids = np.flatnonzero(dones_buffer[step] > 0.0)
            if done_world_ids.size:
                self.reset_worlds(done_world_ids)
            current_obs = self.observe()

        obs_torch = {
            "state": torch.as_tensor(current_obs["state"], dtype=torch.float32, device=device)
        }
        with torch.no_grad():
            next_values = policy.value(obs_torch).detach().cpu().numpy().astype(np.float32)

        return {
            "obs": obs_buffer,
            "actions": actions_buffer,
            "old_log_probs": log_probs_buffer,
            "old_values": values_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "discounts": discounts_buffer,
            "next_values": next_values,
        }


def save_rollout_npz(path: str | os.PathLike[str], rollout: Dict[str, Any]) -> None:
    flat = {}
    for key, value in rollout.items():
        if key == "obs":
            for obs_key, obs_value in value.items():
                flat[f"obs__{obs_key}"] = obs_value
        else:
            flat[key] = value
    np.savez(path, **flat)


def load_rollout_npz(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with np.load(path) as data:
        rollout: Dict[str, Any] = {"obs": {}}
        for key in data.files:
            value = data[key].astype(np.float32, copy=False)
            if key.startswith("obs__"):
                rollout["obs"][key[len("obs__") :]] = value
            else:
                rollout[key] = value
    return rollout


def merge_rollouts(rollouts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rollout_list = list(rollouts)
    if not rollout_list:
        raise ValueError("cannot merge an empty rollout list")

    merged: Dict[str, Any] = {"obs": {}}
    obs_keys = rollout_list[0]["obs"].keys()
    for key in obs_keys:
        merged["obs"][key] = np.concatenate([r["obs"][key] for r in rollout_list], axis=1)

    for key in ("chains", "actions", "old_log_probs", "old_values", "rewards", "dones", "discounts"):
        if key not in rollout_list[0]:
            continue
        merged[key] = np.concatenate([r[key] for r in rollout_list], axis=1)
    merged["next_values"] = np.concatenate([r["next_values"] for r in rollout_list], axis=0)
    return merged


def finish_rollout_advantages(
    rollout: Dict[str, Any],
    gamma: float,
    gae_lambda: float,
    reward_normalizer=None,
) -> Dict[str, Any]:
    rewards = np.nan_to_num(
        rollout["rewards"].astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if reward_normalizer is not None:
        rewards = reward_normalizer.normalize(rewards).astype(np.float32)
        rollout["rewards"] = rewards

    values = np.nan_to_num(
        rollout["old_values"].astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    discounts = np.nan_to_num(
        rollout["discounts"].astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    next_values = np.nan_to_num(
        rollout["next_values"].astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    n_steps, n_envs = rewards.shape

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = np.zeros(n_envs, dtype=np.float32)
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_values_step = next_values
        else:
            next_values_step = values[step + 1]
        delta = rewards[step] + discounts[step] * next_values_step - values[step]
        last_gae_lam = delta + discounts[step] * float(gae_lambda) * last_gae_lam
        advantages[step] = last_gae_lam

    rollout["advantages"] = advantages
    rollout["returns"] = advantages + values
    return rollout
