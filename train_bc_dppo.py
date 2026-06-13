import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def preparse_runtime_env() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--mujoco_gl", type=str, default=None)
    args, _ = parser.parse_known_args()

    target_gpu = str(args.gpu_idx)
    default_gl = "glfw" if os.name == "nt" else "egl"
    mujoco_gl = args.mujoco_gl or default_gl

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", target_gpu)
    os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    if mujoco_gl == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", target_gpu)
    return target_gpu


TARGET_GPU = preparse_runtime_env()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    import h5py
except ImportError:
    h5py = None
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
try:
    import swanlab
except ImportError:
    swanlab = None

from RLenv import RLEnv
from models.dppo_mae import DPPOPolicy, observation_to_eval_frame, save_gif
from models.pretrain_models import VTT, VTMAE

DEFAULT_ACTION_HORIZON = 8
DEFAULT_ACTION_EXEC_STEPS = 4
STAGE_NAMES = ("reaching", "near_object", "stable_grasp", "near_goal")
STAGE_KEYWORDS = {
    "reaching": ("reaching", "reach"),
    "near_object": ("near_object", "near_obj", "naer_object", "naer_obj", "pregrasp", "pre_grasp"),
    "stable_grasp": ("stable_grasp", "stalbe_grasp"),
    "near_goal": ("near_goal", "neargoal", "place", "placing"),
}


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"boolean argument should be either True or False (got {value})")


def parse_stage_ratios(value: str) -> Tuple[float, float, float, float]:
    normalized = (
        str(value)
        .replace("：", ":")
        .replace(",", ":")
        .replace("，", ":")
        .replace(";", ":")
        .replace("；", ":")
    )
    parts = [part.strip() for part in normalized.split(":") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--stage_ratios must contain four values for "
            "reaching:near_object:stable_grasp:near_goal, e.g. 0:0:0.5:0.5"
        )

    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--stage_ratios values must be numeric") from exc

    if any((not math.isfinite(ratio)) or ratio < 0.0 for ratio in ratios):
        raise argparse.ArgumentTypeError("--stage_ratios values must be non-negative finite numbers")
    if sum(ratios) <= 0.0:
        raise argparse.ArgumentTypeError("--stage_ratios must have at least one positive value")
    return ratios  # type: ignore[return-value]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def close_env(env: Optional[RLEnv]) -> None:
    if env is None:
        return
    renderer = getattr(env, "renderer", None)
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass


def make_env(config, seed: int) -> RLEnv:
    return RLEnv(
        xml_path=config.xml_path,
        frame_stack=config.frame_stack,
        ur=config.ur,
        img_size=config.img_size,
        tactile_size=config.tactile_size,
        max_steps=config.eval_env_max_steps,
        reset_types=config.eval_reset_types,
        success_distance_threshold=config.eval_success_distance_threshold,
        reset_bank_size_per_type=config.reset_bank_size_per_type,
        reset_cache_dir=config.reset_cache_dir,
        reset_force_rebuild=config.reset_force_rebuild,
        reset_bank_verbose=config.reset_bank_verbose,
        reset_randomize_on_reset=config.reset_randomize_on_reset,
    )


def reset_env_with_retries(
    env: RLEnv,
    seed: int,
    max_attempts: int,
    seed_stride: int,
    reset_options: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], int]:
    max_attempts = max(int(max_attempts), 1)
    seed_stride = max(int(seed_stride), 1)
    last_error: Optional[Exception] = None
    for attempt_idx in range(max_attempts):
        candidate_seed = int(seed + attempt_idx * seed_stride)
        try:
            observation, reset_info = env.reset(seed=candidate_seed, options=reset_options)
            if attempt_idx > 0:
                print(
                    f"[BC] eval reset recovered: requested_seed={seed} "
                    f"used_seed={candidate_seed} attempts={attempt_idx + 1}"
                )
            return observation, dict(reset_info), candidate_seed
        except RuntimeError as exc:
            last_error = exc
            if "failed to sample a valid OmniReset state" not in str(exc):
                raise
            print(
                f"[BC] eval reset failed for seed={candidate_seed}; "
                f"retrying ({attempt_idx + 1}/{max_attempts})"
            )
    raise RuntimeError(
        f"failed to reset eval env after {max_attempts} attempts "
        f"(first_seed={seed}, seed_stride={seed_stride})"
    ) from last_error


def build_mae(config, device: torch.device) -> VTMAE:
    num_tactiles = 2
    encoder = VTT(
        image_size=(config.img_size, config.img_size),
        tactile_size=(config.tactile_size, config.tactile_size),
        image_patch_size=(8, 8),
        tactile_patch_size=(4, 4),
        dim=config.dim_embedding,
        depth=4,
        heads=4,
        mlp_dim=config.dim_embedding * 2,
        num_tactiles=num_tactiles,
        image_channels=6 * config.frame_stack,
        tactile_channels=3 * config.frame_stack,
        frame_stack=config.frame_stack,
    )
    mae = VTMAE(
        encoder=encoder,
        masking_ratio=config.masking_ratio,
        decoder_dim=config.dim_embedding,
        decoder_depth=3,
        decoder_heads=4,
        num_tactiles=num_tactiles,
        early_conv_masking=True,
        use_sincosmod_encodings=True,
        frame_stack=config.frame_stack,
    )
    mae.initialize_training({"lr": config.learning_rate, "batch_size": config.batch_size})
    return mae.to(device)


def extract_state_dict(
    checkpoint_obj,
    preferred_keys: Sequence[str],
    strip_prefixes: Sequence[str] = (),
) -> Dict[str, torch.Tensor]:
    state_dict = None
    if isinstance(checkpoint_obj, dict):
        for key in preferred_keys:
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
        if state_dict is None and all(isinstance(key, str) for key in checkpoint_obj.keys()):
            state_dict = checkpoint_obj
    if state_dict is None:
        raise ValueError("unable to locate a state_dict in checkpoint")

    for prefix in strip_prefixes:
        stripped = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if stripped:
            state_dict = stripped
            break

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    return state_dict


def load_policy_checkpoint(policy: DPPOPolicy, checkpoint_path: str, strict: bool = False) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint, preferred_keys=("policy_state_dict", "state_dict"))
    incompatible = policy.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        f"[BC] loaded policy checkpoint from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[BC] first missing keys: {missing[:10]}")
    if unexpected:
        print(f"[BC] first unexpected keys: {unexpected[:10]}")


def load_mae_checkpoint(mae_model: VTMAE, checkpoint_path: str, strict: bool = False) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(
        checkpoint,
        preferred_keys=("mae_state_dict", "state_dict", "model_state_dict", "model", "mae", "policy_state_dict"),
        strip_prefixes=("encoder.mae_model.", "mae_model."),
    )
    incompatible = mae_model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        f"[BC] loaded MAE checkpoint from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[BC] first missing keys: {missing[:10]}")
    if unexpected:
        print(f"[BC] first unexpected keys: {unexpected[:10]}")


def init_swanlab(config):
    if not config.use_swanlab:
        return None
    if swanlab is None:
        print("[BC] swanlab is not installed, logging disabled")
        return None

    init_kwargs = {
        "project": config.project_name,
        "experiment_name": config.experiment_name,
        "config": vars(config),
        "mode": config.swanlab_mode,
    }
    if config.swanlab_workspace:
        init_kwargs["workspace"] = config.swanlab_workspace

    try:
        swanlab.init(**init_kwargs)
        print(
            f"[BC] swanlab enabled "
            f"(project={config.project_name}, experiment={config.experiment_name}, mode={config.swanlab_mode})"
        )
        return swanlab
    except Exception as exc:
        print(f"[BC] swanlab init failed, logging disabled: {exc}")
        return None


def log_to_swanlab(swanlab_module, metrics: Dict, step: int) -> None:
    if swanlab_module is None:
        return
    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool, np.integer, np.floating))
    }
    swanlab_module.log(scalar_metrics, step=step)


def wrap_progress(iterable, enabled: bool, **kwargs):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def set_progress_postfix(progress, **kwargs) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(**kwargs)


def close_progress(progress) -> None:
    if hasattr(progress, "close"):
        progress.close()


def _append_files_from_dirs(
    files: List[Path],
    seen: set[str],
    data_dirs: Sequence[str],
    suffixes: Sequence[str],
) -> None:
    for data_dir in data_dirs:
        root = Path(data_dir).expanduser()
        for suffix in suffixes:
            for path in sorted(root.glob(f"*{suffix}")):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(path)


def discover_candidate_episode_paths(
    data_dirs: Sequence[str],
    read_depth_data_dirs: Optional[Sequence[str]] = None,
    keyboard_data_dirs: Optional[Sequence[str]] = None,
) -> List[Path]:
    files: List[Path] = []
    seen: set[str] = set()
    _append_files_from_dirs(files, seen, data_dirs, (".npz", ".hdf5", ".h5"))
    _append_files_from_dirs(files, seen, read_depth_data_dirs or (), (".hdf5", ".h5"))
    _append_files_from_dirs(files, seen, keyboard_data_dirs or (), (".npz",))
    return files


@dataclass
class EpisodeSpec:
    path: Path
    file_format: str
    num_steps: int
    action_dim: int
    has_tactile: bool
    image_shape: Tuple[int, ...]
    tactile_shape: Optional[Tuple[int, ...]]
    pre_stacked: bool


class ExpertSequenceDataset(Dataset):
    def __init__(
        self,
        episode_specs: Sequence[EpisodeSpec],
        action_horizon: int,
        vision_only_control: bool,
        frame_stack: int,
        img_size: int,
        tactile_size: int,
        ur: bool,
        ee_pos_step: float,
        ee_rot_step: float,
        gripper_ctrl_step: float,
        hdf5_gripper_min: float,
        hdf5_gripper_max: float,
        sample_stride: int = 1,
    ) -> None:
        self.episode_specs = list(episode_specs)
        self.action_horizon = action_horizon
        self.vision_only_control = vision_only_control
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.tactile_size = int(tactile_size)
        self.ur = bool(ur)
        self.ee_pos_step = float(ee_pos_step)
        self.ee_rot_step = float(ee_rot_step)
        self.gripper_ctrl_step = float(gripper_ctrl_step)
        self.hdf5_gripper_min = float(hdf5_gripper_min)
        self.hdf5_gripper_max = float(hdf5_gripper_max)
        self.sample_stride = max(int(sample_stride), 1)
        self.samples: List[Tuple[int, int]] = []
        for episode_idx, spec in enumerate(self.episode_specs):
            if spec.pre_stacked:
                start_idx = 0
                end_idx = spec.num_steps - self.action_horizon + 1
            else:
                start_idx = max(self.frame_stack - 1, 0)
                end_idx = spec.num_steps - self.action_horizon
            for step_idx in range(start_idx, max(end_idx, start_idx), self.sample_stride):
                self.samples.append((episode_idx, step_idx))
        self._cache_episode_idx: Optional[int] = None
        self._cache_data: Optional[Dict[str, object]] = None
        self._hdf5_handles: Dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache_episode_idx"] = None
        state["_cache_data"] = None
        state["_hdf5_handles"] = {}
        return state

    def _close_hdf5_handles(self) -> None:
        for handle in self._hdf5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._hdf5_handles.clear()

    def __del__(self):
        self._close_hdf5_handles()

    def _get_hdf5_handle(self, episode_idx: int):
        handle = self._hdf5_handles.get(episode_idx)
        if handle is None:
            spec = self.episode_specs[episode_idx]
            handle = h5py.File(spec.path, "r")
            self._hdf5_handles[episode_idx] = handle
        return handle

    def _load_episode(self, episode_idx: int) -> Dict[str, object]:
        if self._cache_episode_idx == episode_idx and self._cache_data is not None:
            return self._cache_data

        spec = self.episode_specs[episode_idx]
        if spec.file_format == "npz":
            with np.load(spec.path, allow_pickle=True) as episode:
                images = np.asarray(episode["images"], dtype=np.uint8)
                actions = np.asarray(episode["actions"], dtype=np.float32)
                payload = {
                    "images": images,
                    "actions": actions,
                }
                if spec.has_tactile and not self.vision_only_control:
                    payload["tactiles"] = np.asarray(episode["tactiles"], dtype=np.uint8)
        elif spec.file_format == "hdf5":
            if h5py is None:
                raise ImportError("h5py is required to load hdf5 expert data")
            episode = self._get_hdf5_handle(episode_idx)
            if spec.pre_stacked:
                payload = {
                    "images": episode["images"],
                    "actions": episode["actions"],
                }
                if spec.has_tactile and not self.vision_only_control:
                    payload["tactiles"] = episode["tactiles"]
            else:
                payload = {
                    "eye_in_hand": episode["eye_in_hand"],
                    "eye_to_hand": episode["eye_to_hand"],
                    "pos": episode["pos"],
                    "griper": episode["griper"] if "griper" in episode else episode["gripper"],
                }
                if spec.has_tactile and not self.vision_only_control:
                    payload["tactile"] = episode["tactile"]
        else:
            raise ValueError(f"unsupported episode format '{spec.file_format}'")

        self._cache_episode_idx = episode_idx
        self._cache_data = payload
        return payload

    def _resize_hwc(self, image: np.ndarray, size: int) -> np.ndarray:
        if image.shape[0] == size and image.shape[1] == size:
            return image
        tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        resized = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(np.rint(resized), 0, 255).astype(image.dtype)

    def _resize_chw(self, image: np.ndarray, size: int) -> np.ndarray:
        if image.shape[-2] == size and image.shape[-1] == size:
            return image
        tensor = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        resized = tensor.squeeze(0).cpu().numpy()
        return np.clip(np.rint(resized), 0, 255).astype(image.dtype)

    def _wrap_angle_delta(self, delta: np.ndarray) -> np.ndarray:
        return (delta + np.pi) % (2.0 * np.pi) - np.pi

    def _normalize_gripper(self, value: np.ndarray) -> np.ndarray:
        scale = max(self.hdf5_gripper_max - self.hdf5_gripper_min, 1e-6)
        normalized = 2.0 * (value - self.hdf5_gripper_min) / scale - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def _build_hdf5_observation(
        self,
        payload: Dict[str, object],
        step_idx: int,
    ) -> Dict[str, torch.Tensor]:
        start_idx = step_idx - self.frame_stack + 1
        end_idx = step_idx + 1
        eye_in_hand = np.asarray(payload["eye_in_hand"][start_idx:end_idx], dtype=np.uint8)
        eye_to_hand = np.asarray(payload["eye_to_hand"][start_idx:end_idx], dtype=np.uint8)

        image_frames = []
        for frame_in_hand, frame_to_hand in zip(eye_in_hand, eye_to_hand):
            if frame_in_hand.shape[:2] != (self.img_size, self.img_size):
                frame_in_hand = self._resize_hwc(frame_in_hand, self.img_size)
            if frame_to_hand.shape[:2] != (self.img_size, self.img_size):
                frame_to_hand = self._resize_hwc(frame_to_hand, self.img_size)
            image_frames.append(np.concatenate((frame_in_hand, frame_to_hand), axis=-1))

        observation = {
            "image": torch.from_numpy(np.concatenate(image_frames, axis=-1)),
        }

        if "tactile" in payload:
            tactile_window = np.asarray(payload["tactile"][start_idx:end_idx], dtype=np.uint8)
            stacked_tactile = tactile_window.reshape(
                -1,
                tactile_window.shape[-2],
                tactile_window.shape[-1],
            )
            if stacked_tactile.shape[-2:] != (self.tactile_size, self.tactile_size):
                stacked_tactile = self._resize_chw(stacked_tactile, self.tactile_size)
            observation["tactile"] = torch.from_numpy(stacked_tactile)

        return observation

    def _build_hdf5_actions(
        self,
        payload: Dict[str, object],
        step_idx: int,
    ) -> torch.Tensor:
        end_idx = step_idx + self.action_horizon + 1
        pose_seq = np.asarray(payload["pos"][step_idx:end_idx], dtype=np.float32)
        gripper_seq = np.asarray(payload["griper"][step_idx:end_idx], dtype=np.float32).reshape(-1)
        actions = []
        for offset in range(self.action_horizon):
            cur_pose = pose_seq[offset]
            next_pose = pose_seq[offset + 1]

            delta_pos = (next_pose[:3] - cur_pose[:3]) / max(self.ee_pos_step, 1e-6)
            delta_rot = self._wrap_angle_delta(next_pose[3:6] - cur_pose[3:6]) / max(self.ee_rot_step, 1e-6)

            cur_gripper = self._normalize_gripper(gripper_seq[offset])
            next_gripper = self._normalize_gripper(gripper_seq[offset + 1])
            delta_gripper = (next_gripper - cur_gripper) / max(self.gripper_ctrl_step, 1e-6)

            if self.ur:
                action = np.concatenate((delta_pos, delta_rot, np.array([delta_gripper], dtype=np.float32)))
            else:
                action = np.array(
                    [
                        delta_pos[0],
                        delta_pos[1],
                        delta_pos[2],
                        delta_rot[2],
                        delta_gripper,
                    ],
                    dtype=np.float32,
                )
            actions.append(np.clip(action, -1.0, 1.0).astype(np.float32))
        return torch.from_numpy(np.stack(actions, axis=0))

    def __getitem__(self, index: int):
        episode_idx, step_idx = self.samples[index]
        payload = self._load_episode(episode_idx)
        spec = self.episode_specs[episode_idx]

        if spec.file_format == "hdf5" and not spec.pre_stacked:
            observations = self._build_hdf5_observation(payload, step_idx)
            actions = self._build_hdf5_actions(payload, step_idx)
            return observations, actions

        observations = {
            "image": torch.from_numpy(payload["images"][step_idx]),
        }
        if "tactiles" in payload:
            observations["tactile"] = torch.from_numpy(payload["tactiles"][step_idx])

        actions = torch.from_numpy(
            payload["actions"][step_idx : step_idx + self.action_horizon]
        ).float()
        return observations, actions


def discover_episode_specs(
    data_dirs: Sequence[str],
    frame_stack: int,
    action_horizon: int,
    require_tactile: bool,
    ur: bool,
    read_depth_data_dirs: Optional[Sequence[str]] = None,
    keyboard_data_dirs: Optional[Sequence[str]] = None,
    candidate_paths: Optional[Sequence[Path]] = None,
) -> Tuple[List[EpisodeSpec], List[Tuple[Path, str]]]:
    episode_specs: List[EpisodeSpec] = []
    skipped: List[Tuple[Path, str]] = []
    expected_image_channels = 6 * frame_stack

    files = (
        [Path(path) for path in candidate_paths]
        if candidate_paths is not None
        else discover_candidate_episode_paths(data_dirs, read_depth_data_dirs, keyboard_data_dirs)
    )

    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".npz":
                with np.load(path, allow_pickle=True) as episode:
                    if "images" not in episode or "actions" not in episode:
                        skipped.append((path, "missing images/actions"))
                        continue

                    images = np.asarray(episode["images"], dtype=np.uint8)
                    actions = np.asarray(episode["actions"], dtype=np.float32)
                    if actions.ndim == 1:
                        actions = actions[:, None]

                    if images.ndim != 4:
                        skipped.append((path, f"images ndim {images.ndim} != 4"))
                        continue
                    if images.shape[0] != actions.shape[0]:
                        skipped.append((path, "images/actions length mismatch"))
                        continue
                    if images.shape[-1] != expected_image_channels:
                        skipped.append(
                            (path, f"image channels {images.shape[-1]} != expected {expected_image_channels}")
                        )
                        continue

                    has_tactile = "tactiles" in episode.files
                    tactile_shape = None
                    if has_tactile:
                        tactiles = np.asarray(episode["tactiles"], dtype=np.uint8)
                        tactile_shape = tuple(tactiles.shape)
                        if tactiles.shape[0] != actions.shape[0]:
                            skipped.append((path, "tactiles/actions length mismatch"))
                            continue
                    elif require_tactile:
                        skipped.append((path, "missing tactiles"))
                        continue

                    if actions.shape[0] < action_horizon:
                        skipped.append((path, "too short for requested action_horizon"))
                        continue

                    episode_specs.append(
                        EpisodeSpec(
                            path=path,
                            file_format="npz",
                            num_steps=int(actions.shape[0]),
                            action_dim=int(actions.shape[1]),
                            has_tactile=has_tactile,
                            image_shape=tuple(images.shape),
                            tactile_shape=tactile_shape,
                            pre_stacked=True,
                        )
                    )
            elif suffix in {".hdf5", ".h5"}:
                if h5py is None:
                    skipped.append((path, "h5py is not installed"))
                    continue
                with h5py.File(path, "r") as episode:
                    if "images" in episode and "actions" in episode:
                        images = episode["images"]
                        actions = episode["actions"]

                        if len(images.shape) != 4:
                            skipped.append((path, f"images ndim {len(images.shape)} != 4"))
                            continue
                        if len(actions.shape) == 1:
                            action_dim = 1
                        elif len(actions.shape) == 2:
                            action_dim = int(actions.shape[1])
                        else:
                            skipped.append((path, f"actions ndim {len(actions.shape)} not in {{1, 2}}"))
                            continue
                        if images.shape[0] != actions.shape[0]:
                            skipped.append((path, "images/actions length mismatch"))
                            continue
                        if images.shape[-1] != expected_image_channels:
                            skipped.append(
                                (path, f"image channels {images.shape[-1]} != expected {expected_image_channels}")
                            )
                            continue

                        has_tactile = "tactiles" in episode.keys()
                        tactile_shape = None
                        if has_tactile:
                            tactiles = episode["tactiles"]
                            tactile_shape = tuple(tactiles.shape)
                            if tactiles.shape[0] != actions.shape[0]:
                                skipped.append((path, "tactiles/actions length mismatch"))
                                continue
                        elif require_tactile:
                            skipped.append((path, "missing tactiles"))
                            continue

                        if actions.shape[0] < action_horizon:
                            skipped.append((path, "too short for requested action_horizon"))
                            continue

                        episode_specs.append(
                            EpisodeSpec(
                                path=path,
                                file_format="hdf5",
                                num_steps=int(actions.shape[0]),
                                action_dim=action_dim,
                                has_tactile=has_tactile,
                                image_shape=tuple(images.shape),
                                tactile_shape=tactile_shape,
                                pre_stacked=True,
                            )
                        )
                        continue

                    if "eye_in_hand" not in episode or "eye_to_hand" not in episode or "pos" not in episode:
                        skipped.append((path, "missing eye_in_hand / eye_to_hand / pos"))
                        continue
                    if "griper" not in episode and "gripper" not in episode:
                        skipped.append((path, "missing griper / gripper"))
                        continue

                    eye_in_hand = episode["eye_in_hand"]
                    eye_to_hand = episode["eye_to_hand"]
                    pos = episode["pos"]
                    gripper = episode["griper"] if "griper" in episode else episode["gripper"]

                    if len(eye_in_hand.shape) != 4 or len(eye_to_hand.shape) != 4:
                        skipped.append((path, "camera datasets must be 4D"))
                        continue
                    if eye_in_hand.shape != eye_to_hand.shape:
                        skipped.append((path, "camera shapes mismatch"))
                        continue
                    if eye_in_hand.shape[-1] != 3 or eye_to_hand.shape[-1] != 3:
                        skipped.append((path, "camera channels must be RGB"))
                        continue
                    if len(pos.shape) != 2 or pos.shape[1] != 6:
                        skipped.append((path, "pos must have shape [T, 6]"))
                        continue
                    if eye_in_hand.shape[0] != pos.shape[0] or gripper.shape[0] != pos.shape[0]:
                        skipped.append((path, "camera/pos/gripper length mismatch"))
                        continue

                    has_tactile = "tactile" in episode.keys()
                    tactile_shape = None
                    if has_tactile:
                        tactile = episode["tactile"]
                        tactile_shape = tuple(tactile.shape)
                        if len(tactile.shape) != 4 or tactile.shape[1] != 6:
                            skipped.append((path, "tactile must have shape [T, 6, H, W]"))
                            continue
                        if tactile.shape[0] != pos.shape[0]:
                            skipped.append((path, "tactile/pos length mismatch"))
                            continue
                    elif require_tactile:
                        skipped.append((path, "missing tactile"))
                        continue

                    if pos.shape[0] < max(frame_stack, 1) + action_horizon:
                        skipped.append((path, "too short for requested frame_stack/action_horizon"))
                        continue

                    episode_specs.append(
                        EpisodeSpec(
                            path=path,
                            file_format="hdf5",
                            num_steps=int(pos.shape[0]),
                            action_dim=7 if ur else 5,
                            has_tactile=has_tactile,
                            image_shape=(
                                int(eye_in_hand.shape[0]),
                                int(eye_in_hand.shape[1]),
                                int(eye_in_hand.shape[2]),
                                6,
                            ),
                            tactile_shape=tactile_shape,
                            pre_stacked=False,
                        )
                    )
            else:
                skipped.append((path, f"unsupported suffix {suffix}"))
        except Exception as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))

    return episode_specs, skipped


def infer_episode_stage(path: Path) -> Optional[str]:
    name = path.name.lower()
    for stage_name in STAGE_NAMES:
        if any(keyword in name for keyword in STAGE_KEYWORDS[stage_name]):
            return stage_name
    return None


def stage_counts_for_specs(episode_specs: Sequence[EpisodeSpec]) -> Dict[str, int]:
    counts = {stage_name: 0 for stage_name in STAGE_NAMES}
    counts["unknown"] = 0
    for spec in episode_specs:
        stage_name = infer_episode_stage(spec.path)
        counts[stage_name if stage_name is not None else "unknown"] += 1
    return counts


def stage_counts_for_paths(paths: Sequence[Path]) -> Dict[str, int]:
    counts = {stage_name: 0 for stage_name in STAGE_NAMES}
    counts["unknown"] = 0
    for path in paths:
        stage_name = infer_episode_stage(Path(path))
        counts[stage_name if stage_name is not None else "unknown"] += 1
    return counts


def format_stage_ratios(stage_ratios: Sequence[float]) -> str:
    return ", ".join(
        f"{stage_name}={float(ratio):g}"
        for stage_name, ratio in zip(STAGE_NAMES, stage_ratios)
    )


def select_episode_specs_by_stage_ratios(
    episode_specs: Sequence[EpisodeSpec],
    stage_ratios: Sequence[float],
    sample_total: int,
    seed: int,
) -> Tuple[List[EpisodeSpec], Dict[str, object]]:
    sample_total = int(sample_total)
    if sample_total < 1:
        raise ValueError("--stage_sample_total must be at least 1")

    ratio_sum = float(sum(stage_ratios))
    if ratio_sum <= 0.0:
        raise ValueError("--stage_ratios must have at least one positive value")
    normalized_ratios = {
        stage_name: float(ratio) / ratio_sum
        for stage_name, ratio in zip(STAGE_NAMES, stage_ratios)
    }

    buckets: Dict[str, List[EpisodeSpec]] = {stage_name: [] for stage_name in STAGE_NAMES}
    unknown_specs: List[EpisodeSpec] = []
    for spec in episode_specs:
        stage_name = infer_episode_stage(spec.path)
        if stage_name is None:
            unknown_specs.append(spec)
        else:
            buckets[stage_name].append(spec)

    active_stages = [
        stage_name for stage_name in STAGE_NAMES if normalized_ratios[stage_name] > 0.0
    ]
    if not active_stages:
        raise ValueError("--stage_ratios must select at least one stage")

    missing_stages = [stage_name for stage_name in active_stages if not buckets[stage_name]]
    if missing_stages:
        raise RuntimeError(
            "stage ratios request data that was not found by filename: "
            + ", ".join(missing_stages)
        )

    rng = random.Random(seed)
    for specs in buckets.values():
        rng.shuffle(specs)

    max_total = min(
        len(buckets[stage_name]) / normalized_ratios[stage_name]
        for stage_name in active_stages
    )
    max_available_total = int(math.floor(max_total + 1e-9))
    if max_available_total < 1:
        raise RuntimeError(
            "not enough episodes to sample from requested --stage_ratios; "
            f"available={stage_counts_for_specs(episode_specs)}"
        )
    if sample_total > max_available_total:
        required_counts = {
            stage_name: int(math.ceil(normalized_ratios[stage_name] * sample_total - 1e-9))
            for stage_name in active_stages
        }
        raise RuntimeError(
            "--stage_sample_total is larger than available data for the requested ratios: "
            f"requested_total={sample_total}, max_available_total={max_available_total}, "
            f"required_counts={required_counts}, "
            f"available_counts={stage_counts_for_specs(episode_specs)}"
        )
    target_total = sample_total

    raw_counts = {
        stage_name: normalized_ratios[stage_name] * target_total
        for stage_name in active_stages
    }
    target_counts = {
        stage_name: int(math.floor(raw_counts[stage_name]))
        for stage_name in active_stages
    }
    remainder = target_total - sum(target_counts.values())
    fractional_order = sorted(
        active_stages,
        key=lambda stage_name: raw_counts[stage_name] - target_counts[stage_name],
        reverse=True,
    )
    while remainder > 0:
        changed = False
        for stage_name in fractional_order:
            if target_counts[stage_name] >= len(buckets[stage_name]):
                continue
            target_counts[stage_name] += 1
            remainder -= 1
            changed = True
            if remainder == 0:
                break
        if not changed:
            break

    selected_specs: List[EpisodeSpec] = []
    selected_counts = {stage_name: 0 for stage_name in STAGE_NAMES}
    for stage_name in STAGE_NAMES:
        count = target_counts.get(stage_name, 0)
        selected = buckets[stage_name][:count]
        selected_specs.extend(selected)
        selected_counts[stage_name] = len(selected)
    rng.shuffle(selected_specs)

    summary = {
        "requested_ratios": {
            stage_name: float(ratio)
            for stage_name, ratio in zip(STAGE_NAMES, stage_ratios)
        },
        "normalized_ratios": normalized_ratios,
        "requested_total": sample_total,
        "max_available_total": max_available_total,
        "available_counts": {
            **{stage_name: len(buckets[stage_name]) for stage_name in STAGE_NAMES},
            "unknown": len(unknown_specs),
        },
        "selected_counts": selected_counts,
        "selected_total": len(selected_specs),
        "unknown_preview": [str(spec.path) for spec in unknown_specs[:8]],
    }
    return selected_specs, summary


def select_episode_paths_by_stage_ratios(
    paths: Sequence[Path],
    stage_ratios: Sequence[float],
    sample_total: int,
    seed: int,
) -> Tuple[List[Path], Dict[str, object]]:
    sample_total = int(sample_total)
    if sample_total < 1:
        raise ValueError("--stage_sample_total must be at least 1")

    ratio_sum = float(sum(stage_ratios))
    if ratio_sum <= 0.0:
        raise ValueError("--stage_ratios must have at least one positive value")
    normalized_ratios = {
        stage_name: float(ratio) / ratio_sum
        for stage_name, ratio in zip(STAGE_NAMES, stage_ratios)
    }

    buckets: Dict[str, List[Path]] = {stage_name: [] for stage_name in STAGE_NAMES}
    unknown_paths: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        stage_name = infer_episode_stage(path)
        if stage_name is None:
            unknown_paths.append(path)
        else:
            buckets[stage_name].append(path)

    active_stages = [
        stage_name for stage_name in STAGE_NAMES if normalized_ratios[stage_name] > 0.0
    ]
    missing_stages = [stage_name for stage_name in active_stages if not buckets[stage_name]]
    if missing_stages:
        raise RuntimeError(
            "stage ratios request data that was not found by filename: "
            + ", ".join(missing_stages)
        )

    max_total = min(
        len(buckets[stage_name]) / normalized_ratios[stage_name]
        for stage_name in active_stages
    )
    max_available_total = int(math.floor(max_total + 1e-9))
    if sample_total > max_available_total:
        required_counts = {
            stage_name: int(math.ceil(normalized_ratios[stage_name] * sample_total - 1e-9))
            for stage_name in active_stages
        }
        raise RuntimeError(
            "--stage_sample_total is larger than available data for the requested ratios: "
            f"requested_total={sample_total}, max_available_total={max_available_total}, "
            f"required_counts={required_counts}, available_counts={stage_counts_for_paths(paths)}"
        )

    rng = random.Random(seed)
    for stage_paths in buckets.values():
        rng.shuffle(stage_paths)

    raw_counts = {
        stage_name: normalized_ratios[stage_name] * sample_total
        for stage_name in active_stages
    }
    target_counts = {
        stage_name: int(math.floor(raw_counts[stage_name]))
        for stage_name in active_stages
    }
    remainder = sample_total - sum(target_counts.values())
    fractional_order = sorted(
        active_stages,
        key=lambda stage_name: raw_counts[stage_name] - target_counts[stage_name],
        reverse=True,
    )
    while remainder > 0:
        changed = False
        for stage_name in fractional_order:
            if target_counts[stage_name] >= len(buckets[stage_name]):
                continue
            target_counts[stage_name] += 1
            remainder -= 1
            changed = True
            if remainder == 0:
                break
        if not changed:
            break

    selected_paths: List[Path] = []
    selected_counts = {stage_name: 0 for stage_name in STAGE_NAMES}
    for stage_name in STAGE_NAMES:
        count = target_counts.get(stage_name, 0)
        selected = buckets[stage_name][:count]
        selected_paths.extend(selected)
        selected_counts[stage_name] = len(selected)
    rng.shuffle(selected_paths)

    summary = {
        "requested_ratios": {
            stage_name: float(ratio)
            for stage_name, ratio in zip(STAGE_NAMES, stage_ratios)
        },
        "normalized_ratios": normalized_ratios,
        "requested_total": sample_total,
        "max_available_total": max_available_total,
        "available_counts": {
            **{stage_name: len(buckets[stage_name]) for stage_name in STAGE_NAMES},
            "unknown": len(unknown_paths),
        },
        "selected_counts": selected_counts,
        "selected_total": len(selected_paths),
        "unknown_preview": [str(path) for path in unknown_paths[:8]],
    }
    return selected_paths, summary


def split_episode_specs(
    episode_specs: Sequence[EpisodeSpec],
    val_ratio: float,
    seed: int,
) -> Tuple[List[EpisodeSpec], List[EpisodeSpec]]:
    if not episode_specs:
        return [], []

    indices = list(range(len(episode_specs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(round(len(indices) * val_ratio))
    if len(indices) > 1 and val_ratio > 0.0:
        n_val = max(n_val, 1)
    if n_val >= len(indices):
        n_val = max(len(indices) - 1, 0)

    val_indices = set(indices[:n_val])
    train_specs = [spec for idx, spec in enumerate(episode_specs) if idx not in val_indices]
    val_specs = [spec for idx, spec in enumerate(episode_specs) if idx in val_indices]
    return train_specs, val_specs


def move_observations_to_device(
    observations: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in observations.items():
        raw_dtype = value.dtype
        value = value.to(device=device, dtype=torch.float32, non_blocking=True)
        if key == "image" and (raw_dtype == torch.uint8 or float(value.max().item()) > 1.5):
            value = value / 255.0
        if key == "tactile" and raw_dtype == torch.uint8:
            value = value / 255.0
        moved[key] = value
    return moved


def prepare_target_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.dim() == 2:
        actions = actions.unsqueeze(1)
    return actions.clamp(-1.0, 1.0)


def diffusion_bc_loss(
    policy: DPPOPolicy,
    observations: Dict[str, torch.Tensor],
    target_actions: torch.Tensor,
) -> torch.Tensor:
    target_actions = prepare_target_actions(target_actions)
    target_flat = target_actions.reshape(target_actions.shape[0], -1)

    features = policy.encode(observations)
    actor_latent = policy.actor_body(features)
    actor = policy.actor

    train_denoising_steps = getattr(actor, "training_denoising_steps", actor.denoising_steps)
    timesteps = torch.randint(
        low=0,
        high=train_denoising_steps,
        size=(target_flat.shape[0],),
        device=target_flat.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target_flat)
    noisy_actions = actor.q_sample(target_flat, timesteps, noise=noise)
    predicted_noise = actor.denoiser(actor_latent, noisy_actions, timesteps)
    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def evaluate_offline(
    policy: DPPOPolicy,
    dataloader: Optional[DataLoader],
    device: torch.device,
    mae_loss_coef: float,
    vision_only_control: bool,
    max_batches: int = 0,
    progress_bar: bool = True,
    progress_desc: str = "BC val",
) -> Optional[Dict[str, float]]:
    if dataloader is None:
        return None

    policy.eval()
    total_bc_loss = 0.0
    total_mae_loss = 0.0
    total_action_mse = 0.0
    total_action_l1 = 0.0
    total_examples = 0

    progress = wrap_progress(
        dataloader,
        progress_bar,
        desc=progress_desc,
        unit="batch",
        leave=False,
    )
    try:
        for batch_idx, (observations, actions) in enumerate(progress):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_obs = move_observations_to_device(observations, device)
            batch_actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_size = batch_actions.shape[0]

            bc_loss = diffusion_bc_loss(policy, batch_obs, batch_actions)
            predicted_actions, _, _, _ = policy.act(batch_obs, deterministic=True)
            target_actions = prepare_target_actions(batch_actions)
            action_mse = F.mse_loss(predicted_actions, target_actions)
            action_l1 = F.l1_loss(predicted_actions, target_actions)

            if mae_loss_coef > 0.0:
                mae_obs = {"image": batch_obs["image"]} if vision_only_control else batch_obs
                mae_loss = policy.encoder.reconstruction_loss(mae_obs)
            else:
                mae_loss = torch.zeros((), device=device)

            total_bc_loss += float(bc_loss.item()) * batch_size
            total_mae_loss += float(mae_loss.item()) * batch_size
            total_action_mse += float(action_mse.item()) * batch_size
            total_action_l1 += float(action_l1.item()) * batch_size
            total_examples += batch_size

            set_progress_postfix(
                progress,
                bc=f"{bc_loss.item():.4f}",
                mse=f"{action_mse.item():.4f}",
            )
    finally:
        close_progress(progress)

    if total_examples == 0:
        return None

    return {
        "bc_loss": total_bc_loss / total_examples,
        "mae_loss": total_mae_loss / total_examples,
        "action_mse": total_action_mse / total_examples,
        "action_l1": total_action_l1 / total_examples,
    }


def finite_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def min_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return None
    return float(np.min(finite_values))


def action_stats(actions: List[np.ndarray]) -> Dict[str, object]:
    if not actions:
        return {
            "count": 0,
            "saturation_fraction": 0.0,
            "gripper_mean": None,
            "gripper_std": None,
        }
    action_array = np.asarray(actions, dtype=np.float32)
    return {
        "count": int(action_array.shape[0]),
        "saturation_fraction": float(np.mean(np.abs(action_array) >= 0.99)),
        "gripper_mean": float(action_array[:, -1].mean()) if action_array.shape[1] else None,
        "gripper_std": float(action_array[:, -1].std()) if action_array.shape[1] else None,
    }


def temporal_ensemble_action(
    predictions: Sequence[Tuple[int, np.ndarray]],
    fallback_action: np.ndarray,
    decay: float,
) -> np.ndarray:
    if not predictions:
        return np.asarray(fallback_action, dtype=np.float32)
    actions = np.asarray([action for _, action in predictions], dtype=np.float32)
    if actions.shape[0] == 1 or decay <= 0.0:
        return actions.mean(axis=0).astype(np.float32)
    ranks = np.arange(actions.shape[0], dtype=np.float32)
    weights = np.exp(decay * (ranks - ranks.max()))
    weights = weights / np.sum(weights)
    return np.average(actions, axis=0, weights=weights).astype(np.float32)


def build_reset_type_summary(episode_records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for record in episode_records:
        reset_type = str(record.get("reset_type", "unknown"))
        bucket = summary.setdefault(
            reset_type,
            {
                "count": 0,
                "success_count": 0,
                "returns": [],
                "min_success_distances": [],
                "final_success_distances": [],
            },
        )
        bucket["count"] += 1
        bucket["success_count"] += int(bool(record.get("success", False)))
        bucket["returns"].append(float(record.get("return", 0.0)))
        bucket["min_success_distances"].append(record.get("min_success_distance"))
        bucket["final_success_distances"].append(record.get("final_success_distance"))

    for bucket in summary.values():
        returns = bucket.pop("returns")
        min_distances = bucket.pop("min_success_distances")
        final_distances = bucket.pop("final_success_distances")
        count = int(bucket["count"])
        bucket["success_rate"] = float(bucket["success_count"] / max(count, 1))
        bucket["mean_return"] = float(np.mean(returns)) if returns else 0.0
        bucket["mean_min_success_distance"] = mean_optional(min_distances)
        bucket["mean_final_success_distance"] = mean_optional(final_distances)
    return summary


def eval_is_better(
    env_metrics: Dict[str, object],
    best_success_rate: float,
    best_min_success_distance: float,
    best_return: float,
) -> bool:
    success_rate = float(env_metrics["success_rate"])
    mean_return = float(env_metrics["mean_return"])
    mean_min_success_distance = env_metrics.get("mean_min_success_distance")
    distance_for_rank = (
        math.inf
        if mean_min_success_distance is None
        else float(mean_min_success_distance)
    )
    eps = 1e-12
    if success_rate > best_success_rate + eps:
        return True
    if abs(success_rate - best_success_rate) <= eps:
        if distance_for_rank < best_min_success_distance - eps:
            return True
        same_distance = (
            (math.isinf(distance_for_rank) and math.isinf(best_min_success_distance))
            or abs(distance_for_rank - best_min_success_distance) <= eps
        )
        if same_distance and mean_return > best_return + eps:
            return True
    return False


@torch.no_grad()
def evaluate_in_env(
    policy: DPPOPolicy,
    eval_env: RLEnv,
    device: torch.device,
    n_episodes: int,
    seed_start: int,
    seed_retry_stride: int,
    reset_retry_attempts: int,
    reset_sample_index_start: Optional[int],
    action_exec_start: int,
    action_exec_steps: int,
    deterministic: bool,
    eval_warmup_steps: int,
    gif_path: Optional[str],
    success_gif_dir: Optional[str],
    success_log_path: Optional[str],
    gif_fps: int,
    gif_max_frames: int,
    temporal_ensemble: bool,
    temporal_ensemble_decay: float,
    progress_bar: bool = True,
    progress_desc: str = "BC env",
) -> Dict[str, object]:
    policy.eval()
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    successes = 0
    capture_eval_gif = bool(gif_path) and gif_max_frames > 0
    eval_gif_saved = False
    best_eval_gif_return = -math.inf
    best_eval_gif_episode: Optional[int] = None
    success_records: List[Dict[str, object]] = []
    episode_records: List[Dict[str, object]] = []
    reset_types_used: List[str] = []

    progress = wrap_progress(
        range(n_episodes),
        progress_bar,
        desc=progress_desc,
        unit="episode",
        leave=False,
    )
    try:
        for episode_idx in progress:
            requested_env_seed = seed_start + episode_idx
            reset_options = None
            if reset_sample_index_start is not None:
                reset_options = {"reset_sample_index": reset_sample_index_start + episode_idx}
            observation, reset_info, env_seed = reset_env_with_retries(
                eval_env,
                seed=requested_env_seed,
                max_attempts=reset_retry_attempts,
                seed_stride=seed_retry_stride,
                reset_options=reset_options,
            )
            if eval_warmup_steps > 0:
                observation = eval_env.stabilize_after_reset(eval_warmup_steps)
            reset_type = str(reset_info.get("reset_type", "unknown"))
            reset_types_used.append(reset_type)
            done = False
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            done_reason = None
            final_info: Dict[str, object] = {}
            success_distances: List[float] = []
            gripper_obj_distances: List[float] = []
            object_tilt_angles: List[float] = []
            object_rotation_angles: List[float] = []
            executed_actions: List[np.ndarray] = []
            temporal_predictions: Dict[int, List[Tuple[int, np.ndarray]]] = {}
            prediction_order = 0
            success_frames = [] if success_gif_dir else None
            eval_gif_frames = [] if capture_eval_gif else None

            if eval_gif_frames is not None:
                frame = observation_to_eval_frame(observation)
                if frame is not None:
                    eval_gif_frames.append(frame)
                else:
                    image = observation.get("image") if isinstance(observation, dict) else None
                    image_shape = None if image is None else tuple(np.asarray(image).shape)
                    print(
                        "[BC] eval GIF frame skipped at reset: "
                        f"episode={episode_idx} image_shape={image_shape}"
                    )
            if success_frames is not None:
                frame = observation_to_eval_frame(observation)
                if frame is not None:
                    success_frames.append(frame)

            while not done:
                obs_batch = {
                    key: torch.as_tensor(value[None])
                    for key, value in observation.items()
                }
                obs_batch = move_observations_to_device(obs_batch, device)
                action_chunk, _, _, _ = policy.act(obs_batch, deterministic=deterministic)
                action_chunk_np = action_chunk.cpu().numpy()[0]
                prediction_order += 1
                if temporal_ensemble:
                    for future_action_idx in range(action_exec_start, policy.action_horizon):
                        target_step = episode_length + future_action_idx - action_exec_start
                        temporal_predictions.setdefault(target_step, []).append(
                            (
                                prediction_order,
                                np.asarray(action_chunk_np[future_action_idx], dtype=np.float32).copy(),
                            )
                        )

                for exec_offset in range(action_exec_steps):
                    action_idx = action_exec_start + exec_offset
                    fallback_action = np.asarray(action_chunk_np[action_idx], dtype=np.float32)
                    if temporal_ensemble:
                        executed_action = temporal_ensemble_action(
                            temporal_predictions.get(episode_length, []),
                            fallback_action,
                            temporal_ensemble_decay,
                        )
                        temporal_predictions.pop(episode_length, None)
                    else:
                        executed_action = fallback_action
                    executed_actions.append(executed_action)
                    observation, reward, terminated, truncated, info = eval_env.step(executed_action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = bool(terminated or truncated)
                    final_info = dict(info)
                    success_distance = finite_float(info.get("success_distance"))
                    if success_distance is not None:
                        success_distances.append(success_distance)
                    dist_gripper_obj = finite_float(info.get("dist_gripper_obj"))
                    if dist_gripper_obj is not None:
                        gripper_obj_distances.append(dist_gripper_obj)
                    object_tilt_angle = finite_float(info.get("object_tilt_angle_rad"))
                    if object_tilt_angle is not None:
                        object_tilt_angles.append(object_tilt_angle)
                    object_rotation_angle = finite_float(info.get("object_rotation_angle_rad"))
                    if object_rotation_angle is not None:
                        object_rotation_angles.append(object_rotation_angle)
                    if terminated:
                        successes += 1
                        episode_success = True
                        done_reason = "success"
                    elif truncated:
                        done_reason = "truncated"
                    if eval_gif_frames is not None and len(eval_gif_frames) < gif_max_frames:
                        frame = observation_to_eval_frame(observation)
                        if frame is not None:
                            eval_gif_frames.append(frame)
                        elif episode_length == 1:
                            image = observation.get("image") if isinstance(observation, dict) else None
                            image_shape = None if image is None else tuple(np.asarray(image).shape)
                            print(
                                "[BC] eval GIF frame skipped after step: "
                                f"episode={episode_idx} image_shape={image_shape}"
                            )
                    if success_frames is not None:
                        frame = observation_to_eval_frame(observation)
                        if frame is not None:
                            success_frames.append(frame)
                    if done:
                        break

            min_success_distance = min_optional(success_distances)
            final_success_distance = finite_float(final_info.get("success_distance"))
            min_dist_gripper_obj = min_optional(gripper_obj_distances)
            final_dist_gripper_obj = finite_float(final_info.get("dist_gripper_obj"))
            min_object_tilt_angle = min_optional(object_tilt_angles)
            final_object_tilt_angle = finite_float(final_info.get("object_tilt_angle_rad"))
            min_object_rotation_angle = min_optional(object_rotation_angles)
            final_object_rotation_angle = finite_float(final_info.get("object_rotation_angle_rad"))
            episode_action_stats = action_stats(executed_actions)
            success_gif_path = None
            if episode_success and success_frames:
                Path(success_gif_dir).mkdir(parents=True, exist_ok=True)
                success_gif_path = str(
                    Path(success_gif_dir)
                    / f"bc_success_seed_{env_seed}_episode_{episode_idx:04d}.gif"
                )
                if save_gif(success_frames, success_gif_path, fps=gif_fps):
                    print(
                        f"[BC] success GIF saved to {success_gif_path} "
                        f"(seed={env_seed}, frames={len(success_frames)})"
                    )
                else:
                    success_gif_path = None

            if episode_success:
                success_records.append(
                    {
                        "episode": episode_idx,
                        "requested_seed": requested_env_seed,
                        "seed": env_seed,
                        "reset_type": reset_type,
                        "eval_warmup_steps": eval_warmup_steps,
                        "return": episode_return,
                        "length": episode_length,
                        "min_success_distance": min_success_distance,
                        "final_success_distance": final_success_distance,
                        "min_object_tilt_angle_rad": min_object_tilt_angle,
                        "final_object_tilt_angle_rad": final_object_tilt_angle,
                        "gif_path": success_gif_path,
                    }
                )
            episode_records.append(
                {
                    "episode": episode_idx,
                    "requested_seed": requested_env_seed,
                    "seed": env_seed,
                    "reset_type": reset_type,
                    "return": episode_return,
                    "length": episode_length,
                    "success": episode_success,
                    "done_reason": done_reason or "unknown",
                    "min_success_distance": min_success_distance,
                    "final_success_distance": final_success_distance,
                    "min_dist_gripper_obj": min_dist_gripper_obj,
                    "final_dist_gripper_obj": final_dist_gripper_obj,
                    "min_object_tilt_angle_rad": min_object_tilt_angle,
                    "final_object_tilt_angle_rad": final_object_tilt_angle,
                    "min_object_rotation_angle_rad": min_object_rotation_angle,
                    "final_object_rotation_angle_rad": final_object_rotation_angle,
                    "action_stats": episode_action_stats,
                }
            )
            print(
                "[BC] eval episode "
                f"idx={episode_idx} seed={env_seed} reset={reset_type} "
                f"success={episode_success} done={done_reason or 'unknown'} "
                f"return={episode_return:.3f} length={episode_length} "
                f"min_dist={'n/a' if min_success_distance is None else f'{min_success_distance:.5f}'} "
                f"successes={successes}/{episode_idx + 1}"
            )
            if eval_gif_frames is not None and gif_path and (
                not eval_gif_saved or episode_return > best_eval_gif_return
            ):
                if not eval_gif_frames:
                    print(f"[BC] eval GIF not saved: no frames captured for {gif_path}")
                elif save_gif(eval_gif_frames, gif_path, fps=gif_fps):
                    action = "saved" if not eval_gif_saved else "updated"
                    eval_gif_saved = True
                    best_eval_gif_return = float(episode_return)
                    best_eval_gif_episode = int(episode_idx)
                    print(
                        f"[BC] eval GIF {action} to {gif_path} "
                        f"(episode={episode_idx}, seed={env_seed}, return={episode_return:.3f})"
                    )
                else:
                    print(f"[BC] eval GIF not saved: save_gif returned False for {gif_path}")
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            set_progress_postfix(
                progress,
                ret=f"{episode_return:.2f}",
                seed=env_seed,
                reset=reset_type,
                min_dist="n/a" if min_success_distance is None else f"{min_success_distance:.4f}",
            )
    finally:
        close_progress(progress)

    if capture_eval_gif and gif_path:
        if eval_gif_saved:
            print(
                f"[BC] best eval GIF return={best_eval_gif_return:.3f} "
                f"episode={best_eval_gif_episode} path={gif_path}"
            )
        else:
            print(f"[BC] eval GIF not saved: no episode produced frames for {gif_path}")

    if success_log_path:
        Path(success_log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(success_log_path).open("w", encoding="utf-8") as handle:
            json.dump(success_records, handle, indent=2, ensure_ascii=False)

    return {
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "success_rate": float(successes / max(n_episodes, 1)),
        "success_count": successes,
        "seed_start": seed_start,
        "seed_end": seed_start + max(n_episodes - 1, 0),
        "episode_seeds": [record["seed"] for record in episode_records],
        "mean_min_success_distance": mean_optional(
            [record["min_success_distance"] for record in episode_records]
        ),
        "mean_final_success_distance": mean_optional(
            [record["final_success_distance"] for record in episode_records]
        ),
        "mean_min_dist_gripper_obj": mean_optional(
            [record["min_dist_gripper_obj"] for record in episode_records]
        ),
        "mean_final_dist_gripper_obj": mean_optional(
            [record["final_dist_gripper_obj"] for record in episode_records]
        ),
        "mean_min_object_tilt_angle_rad": mean_optional(
            [record.get("min_object_tilt_angle_rad") for record in episode_records]
        ),
        "mean_final_object_tilt_angle_rad": mean_optional(
            [record.get("final_object_tilt_angle_rad") for record in episode_records]
        ),
        "mean_min_object_rotation_angle_rad": mean_optional(
            [record.get("min_object_rotation_angle_rad") for record in episode_records]
        ),
        "mean_final_object_rotation_angle_rad": mean_optional(
            [record.get("final_object_rotation_angle_rad") for record in episode_records]
        ),
        "mean_action_saturation_fraction": mean_optional(
            [record["action_stats"]["saturation_fraction"] for record in episode_records]
        ),
        "relaxed_success_rates": {
            str(threshold): float(
                np.mean(
                    [
                        record["min_success_distance"] is not None
                        and record["min_success_distance"] < threshold
                        for record in episode_records
                    ]
                )
            ) if episode_records else 0.0
            for threshold in (2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2)
        },
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_length": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_length": int(np.max(episode_lengths)) if episode_lengths else 0,
        "eval_warmup_steps": eval_warmup_steps,
        "episode_lengths": episode_lengths,
        "episode_records": episode_records,
        "reset_types": reset_types_used,
        "reset_type_summary": build_reset_type_summary(episode_records),
        "success_seeds": [record["seed"] for record in success_records],
        "success_gif_paths": [
            record["gif_path"] for record in success_records if record["gif_path"]
        ],
    }


def aggregate_env_metric_runs(env_metric_runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    episode_records: List[Dict[str, object]] = []
    reset_types_used: List[str] = []
    success_gif_paths: List[str] = []
    repeat_seed_ranges: List[Dict[str, int]] = []
    for run_idx, metrics in enumerate(env_metric_runs):
        episode_records.extend(metrics.get("episode_records", []))
        reset_types_used.extend(metrics.get("reset_types", []))
        success_gif_paths.extend(metrics.get("success_gif_paths", []))
        repeat_seed_ranges.append(
            {
                "repeat": run_idx,
                "seed_start": int(metrics.get("seed_start", 0)),
                "seed_end": int(metrics.get("seed_end", 0)),
            }
        )

    episode_returns = [float(record.get("return", 0.0)) for record in episode_records]
    episode_lengths = [int(record.get("length", 0)) for record in episode_records]
    success_count = int(sum(bool(record.get("success", False)) for record in episode_records))
    episode_seeds = [int(record["seed"]) for record in episode_records if "seed" in record]
    if len(episode_seeds) != len(set(episode_seeds)):
        raise RuntimeError(f"duplicate eval seeds detected: {episode_seeds}")

    n_episodes = len(episode_records)
    return {
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "success_rate": float(success_count / max(n_episodes, 1)),
        "success_count": success_count,
        "seed_start": min(episode_seeds) if episode_seeds else 0,
        "seed_end": max(episode_seeds) if episode_seeds else 0,
        "episode_seeds": episode_seeds,
        "repeat_seed_ranges": repeat_seed_ranges,
        "mean_min_success_distance": mean_optional(
            [record["min_success_distance"] for record in episode_records]
        ),
        "mean_final_success_distance": mean_optional(
            [record["final_success_distance"] for record in episode_records]
        ),
        "mean_min_dist_gripper_obj": mean_optional(
            [record["min_dist_gripper_obj"] for record in episode_records]
        ),
        "mean_final_dist_gripper_obj": mean_optional(
            [record["final_dist_gripper_obj"] for record in episode_records]
        ),
        "mean_min_object_tilt_angle_rad": mean_optional(
            [record.get("min_object_tilt_angle_rad") for record in episode_records]
        ),
        "mean_final_object_tilt_angle_rad": mean_optional(
            [record.get("final_object_tilt_angle_rad") for record in episode_records]
        ),
        "mean_min_object_rotation_angle_rad": mean_optional(
            [record.get("min_object_rotation_angle_rad") for record in episode_records]
        ),
        "mean_final_object_rotation_angle_rad": mean_optional(
            [record.get("final_object_rotation_angle_rad") for record in episode_records]
        ),
        "mean_action_saturation_fraction": mean_optional(
            [record["action_stats"]["saturation_fraction"] for record in episode_records]
        ),
        "relaxed_success_rates": {
            str(threshold): float(
                np.mean(
                    [
                        record["min_success_distance"] is not None
                        and record["min_success_distance"] < threshold
                        for record in episode_records
                    ]
                )
            ) if episode_records else 0.0
            for threshold in (2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2)
        },
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_length": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_length": int(np.max(episode_lengths)) if episode_lengths else 0,
        "eval_warmup_steps": (
            env_metric_runs[0].get("eval_warmup_steps", 0) if env_metric_runs else 0
        ),
        "episode_lengths": episode_lengths,
        "episode_records": episode_records,
        "reset_types": reset_types_used,
        "reset_type_summary": build_reset_type_summary(episode_records),
        "success_seeds": [
            int(record["seed"]) for record in episode_records if bool(record.get("success", False))
        ],
        "success_gif_paths": success_gif_paths,
        "repeat_metrics": list(env_metric_runs),
    }


def resolve_eval_seed_base(config, total_eval_episodes: int) -> int:
    if config.eval_seed_base is not None:
        return int(config.eval_seed_base)
    return 1000


def evaluate_in_env_repeats(
    policy: DPPOPolicy,
    eval_env: RLEnv,
    device: torch.device,
    config,
    eval_seed_base: int,
    tag: str,
    progress_desc: str,
) -> Dict[str, object]:
    repeat_count = max(int(config.eval_repeats), 1)
    total_eval_episodes = max(config.n_eval_episodes * repeat_count, 1)
    repeat_metrics: List[Dict[str, object]] = []
    for repeat_idx in range(repeat_count):
        seed_start = eval_seed_base + repeat_idx * config.n_eval_episodes
        repeat_suffix = f"{tag}_repeat_{repeat_idx + 1:02d}" if repeat_count > 1 else tag
        gif_path = None
        if config.save_eval_gif:
            gif_path = str(Path(config.eval_gif_dir) / f"bc_eval_{repeat_suffix}.gif")
        success_gif_dir = (
            str(Path(config.eval_gif_dir) / f"success_{repeat_suffix}")
            if config.save_success_eval_gif
            else None
        )
        success_log_path = str(Path(config.eval_gif_dir) / f"bc_eval_success_{repeat_suffix}.json")
        desc = (
            f"{progress_desc} repeat {repeat_idx + 1}/{repeat_count}"
            if repeat_count > 1
            else progress_desc
        )
        repeat_metrics.append(
            evaluate_in_env(
                policy=policy,
                eval_env=eval_env,
                device=device,
                n_episodes=config.n_eval_episodes,
                seed_start=seed_start,
                seed_retry_stride=total_eval_episodes,
                reset_retry_attempts=config.eval_reset_retry_attempts,
                reset_sample_index_start=(
                    repeat_idx * config.n_eval_episodes
                    if config.eval_sequential_reset_bank_indices
                    and config.reset_bank_size_per_type > 0
                    else None
                ),
                action_exec_start=config.action_exec_start,
                action_exec_steps=config.action_exec_steps,
                deterministic=config.eval_deterministic,
                eval_warmup_steps=config.eval_warmup_steps,
                gif_path=gif_path,
                success_gif_dir=success_gif_dir,
                success_log_path=success_log_path,
                gif_fps=config.eval_gif_fps,
                gif_max_frames=config.eval_gif_max_frames,
                temporal_ensemble=config.temporal_ensemble,
                temporal_ensemble_decay=config.temporal_ensemble_decay,
                progress_bar=config.progress_bar,
                progress_desc=desc,
            )
        )

    if len(repeat_metrics) == 1:
        env_metrics = repeat_metrics[0]
        env_metrics["repeat_metrics"] = repeat_metrics
        env_metrics["repeat_seed_ranges"] = [
            {
                "repeat": 0,
                "seed_start": int(env_metrics["seed_start"]),
                "seed_end": int(env_metrics["seed_end"]),
            }
        ]
    else:
        env_metrics = aggregate_env_metric_runs(repeat_metrics)
    env_metrics["eval_repeats"] = repeat_count
    env_metrics["eval_seed_base"] = eval_seed_base
    return env_metrics


def save_checkpoint(
    path: Path,
    policy: DPPOPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Dict,
    ema_policy: Optional[DPPOPolicy] = None,
    extra: Optional[Dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if ema_policy is not None:
        payload["ema_policy_state_dict"] = ema_policy.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def append_metrics(log_path: Path, metrics: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def create_ema_policy(policy: DPPOPolicy) -> DPPOPolicy:
    ema_policy = copy.deepcopy(policy)
    freeze_module(ema_policy)
    ema_policy.eval()
    return ema_policy


@torch.no_grad()
def update_ema_policy(ema_policy: DPPOPolicy, policy: DPPOPolicy, decay: float) -> None:
    ema_state = ema_policy.state_dict()
    policy_state = policy.state_dict()
    for key, value in policy_state.items():
        ema_value = ema_state[key]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_value.copy_(value)


def load_ema_policy_checkpoint(ema_policy: DPPOPolicy, checkpoint_obj) -> None:
    if isinstance(checkpoint_obj, dict) and "ema_policy_state_dict" in checkpoint_obj:
        incompatible = ema_policy.load_state_dict(
            checkpoint_obj["ema_policy_state_dict"],
            strict=False,
        )
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        print(
            f"[BC] loaded EMA policy checkpoint "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing:
            print(f"[BC] first EMA missing keys: {missing[:10]}")
        if unexpected:
            print(f"[BC] first EMA unexpected keys: {unexpected[:10]}")
    else:
        state_dict = extract_state_dict(
            checkpoint_obj,
            preferred_keys=("policy_state_dict", "state_dict"),
        )
        ema_policy.load_state_dict(state_dict, strict=False)
        print("[BC] EMA state not found in checkpoint; initialized EMA from policy")


def train_one_epoch(
    policy: DPPOPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float],
    mae_loss_coef: float,
    vision_only_control: bool,
    ema_policy: Optional[DPPOPolicy] = None,
    ema_decay: float = 0.0,
    max_batches: int = 0,
    progress_bar: bool = True,
    progress_desc: str = "BC train",
) -> Dict[str, float]:
    policy.train()
    total_loss = 0.0
    total_bc_loss = 0.0
    total_mae_loss = 0.0
    total_examples = 0

    progress = wrap_progress(
        dataloader,
        progress_bar,
        desc=progress_desc,
        unit="batch",
        leave=False,
    )
    try:
        for batch_idx, (observations, actions) in enumerate(progress):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_obs = move_observations_to_device(observations, device)
            batch_actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_size = batch_actions.shape[0]

            bc_loss = diffusion_bc_loss(policy, batch_obs, batch_actions)
            if mae_loss_coef > 0.0:
                mae_obs = {"image": batch_obs["image"]} if vision_only_control else batch_obs
                mae_loss = policy.encoder.reconstruction_loss(mae_obs)
            else:
                mae_loss = torch.zeros((), device=device)

            loss = bc_loss + mae_loss_coef * mae_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            if ema_policy is not None:
                update_ema_policy(ema_policy, policy, ema_decay)

            total_loss += float(loss.item()) * batch_size
            total_bc_loss += float(bc_loss.item()) * batch_size
            total_mae_loss += float(mae_loss.item()) * batch_size
            total_examples += batch_size

            set_progress_postfix(
                progress,
                loss=f"{loss.item():.4f}",
                bc=f"{bc_loss.item():.4f}",
            )
    finally:
        close_progress(progress)

    mean_divisor = max(total_examples, 1)
    return {
        "loss": total_loss / mean_divisor,
        "bc_loss": total_bc_loss / mean_divisor,
        "mae_loss": total_mae_loss / mean_divisor,
    }


def parse_args():
    parser = argparse.ArgumentParser("Behavior cloning pretraining for DPPOPolicy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_idx", type=int, default=int(TARGET_GPU))
    parser.add_argument("--mujoco_gl", type=str, default=None)
    parser.add_argument("--data_dirs", type=str, nargs="+", default=["./data"])
    parser.add_argument(
        "--stage_ratios",
        type=parse_stage_ratios,
        default=None,
        help=(
            "Four-stage sampling ratios in order "
            "reaching:near_object:stable_grasp:near_goal, e.g. 0:0:0.5:0.5. "
            "Required for training or offline validation; ignored by env-only eval."
        ),
    )
    parser.add_argument("--stage_sample_total", type=int, default=200)
    parser.add_argument(
        "--read_depth_data_dirs",
        "--read_depth_data_path",
        dest="read_depth_data_dirs",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--keyboard_data_dirs",
        "--keyboard_data_path",
        dest="keyboard_data_dirs",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument("--xml_path", type=str, default="./mjmodel.xml")
    parser.add_argument("--run_name", type=str, default="bc_dppo_pretrain")
    parser.add_argument("--use_swanlab", type=str2bool, default=True)
    parser.add_argument("--project_name", type=str, default="DPPO-Robotics")
    parser.add_argument("--experiment_name", type=str, default="bc_dppo_pretrain")
    parser.add_argument("--swanlab_workspace", type=str, default=None)
    parser.add_argument("--swanlab_mode", type=str, default="cloud")

    parser.add_argument("--ur", type=str2bool, default=True)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--tactile_size", type=int, default=64)
    parser.add_argument("--vision_only_control", type=str2bool, default=False)
    parser.add_argument("--hdf5_ee_pos_step", type=float, default=0.005)
    parser.add_argument("--hdf5_ee_rot_step", type=float, default=0.01)
    parser.add_argument("--hdf5_gripper_ctrl_step", type=float, default=0.1)
    parser.add_argument("--hdf5_gripper_min", type=float, default=0.0)
    parser.add_argument("--hdf5_gripper_max", type=float, default=1000.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--shuffle_train", type=str2bool, default=True)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--env_eval_every_epochs", type=int, default=5)
    parser.add_argument("--eval_only", type=str2bool, default=False)
    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--progress_bar", type=str2bool, default=True)

    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--masking_ratio", type=float, default=0.75)
    parser.add_argument("--mae_loss_coef", type=float, default=0.0)
    parser.add_argument("--freeze_encoder", type=str2bool, default=False)
    parser.add_argument("--freeze_critic", type=str2bool, default=True)

    parser.add_argument("--action_horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--action_exec_start", "--action_exec_index", dest="action_exec_start", type=int, default=0)
    parser.add_argument("--action_exec_steps", type=int, default=None)
    parser.add_argument("--temporal_ensemble", type=str2bool, default=True)
    parser.add_argument("--temporal_ensemble_decay", type=float, default=0.01)
    parser.add_argument("--denoising_steps", "--diffusion_steps", dest="denoising_steps", type=int, default=100)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_time_dim", type=int, default=64)
    parser.add_argument("--diffusion_denoiser_type", type=str, default="transformer", choices=["transformer", "cnn", "mlp"])
    parser.add_argument("--diffusion_transformer_depth", type=int, default=2)
    parser.add_argument("--diffusion_transformer_heads", type=int, default=4)
    parser.add_argument("--diffusion_transformer_dropout", type=float, default=0.0)
    parser.add_argument("--diffusion_cnn_depth", type=int, default=4)
    parser.add_argument("--diffusion_cnn_kernel_size", type=int, default=3)
    parser.add_argument("--diffusion_cnn_dropout", type=float, default=0.0)
    parser.add_argument("--diffusion_beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--diffusion_beta_start", type=float, default=1e-4)
    parser.add_argument("--diffusion_beta_end", type=float, default=2e-2)
    parser.add_argument("--use_ddim", type=str2bool, default=True)
    parser.add_argument("--ddim_steps", "--ft_denoising_steps", dest="ddim_steps", type=int, default=5)
    parser.add_argument("--denoised_clip_value", type=float, default=1.0)
    parser.add_argument("--randn_clip_value", type=float, default=3.0)
    parser.add_argument("--final_action_clip_value", type=float, default=1.0)
    parser.add_argument("--eps_clip_value", type=float, default=None)
    parser.add_argument("--min_sampling_std", "--min_sampling_denoising_std", dest="min_sampling_std", type=float, default=0.04)
    parser.add_argument("--min_logprob_std", "--min_logprob_denoising_std", dest="min_logprob_std", type=float, default=0.04)

    parser.add_argument(
        "--pretrained_mae_path",
        "--pretrained_mae_dict_path",
        dest="pretrained_mae_path",
        type=str,
        default=None,
    )
    parser.add_argument("--pretrained_mae_strict", type=str2bool, default=False)
    parser.add_argument("--pretrained_policy_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--eval_repeats", type=int, default=1)
    parser.add_argument("--eval_seed_base", type=int, default=None)
    parser.add_argument("--eval_env_max_steps", type=int, default=1024)
    parser.add_argument("--eval_reset_retry_attempts", type=int, default=64)
    parser.add_argument("--eval_reset_types", type=str, default="reaching")
    parser.add_argument("--eval_success_distance_threshold", type=float, default=2.5e-3)
    parser.add_argument("--reset_bank_size_per_type", type=int, default=50)
    parser.add_argument("--reset_cache_dir", type=str, default="./reset_banks")
    parser.add_argument("--reset_force_rebuild", type=str2bool, default=False)
    parser.add_argument("--reset_bank_verbose", type=str2bool, default=False)
    parser.add_argument("--reset_randomize_on_reset", type=str2bool, default=None)
    parser.add_argument("--eval_sequential_reset_bank_indices", type=str2bool, default=True)
    parser.add_argument("--eval_warmup_steps", type=int, default=50)
    parser.add_argument("--eval_deterministic", type=str2bool, default=True)
    parser.add_argument("--save_eval_gif", type=str2bool, default=False)
    parser.add_argument("--save_success_eval_gif", type=str2bool, default=True)
    parser.add_argument("--eval_gif_dir", type=str, default="./logs/bc_eval_gifs")
    parser.add_argument("--eval_gif_fps", type=int, default=20)
    parser.add_argument("--eval_gif_max_frames", type=int, default=300)

    parser.add_argument("--checkpoint_dir", type=str, default="./logs/bc_checkpoints")
    parser.add_argument("--best_val_path", type=str, default="./logs/bc_checkpoints/best_val.pt")
    parser.add_argument("--best_eval_path", type=str, default="./logs/bc_checkpoints/best_eval.pt")
    parser.add_argument("--final_model_path", type=str, default="./logs/bc_policy_final.pt")
    parser.add_argument("--log_path", type=str, default="./logs/bc_metrics.jsonl")
    args = parser.parse_args()
    if args.action_horizon < 1:
        parser.error("--action_horizon must be at least 1")
    if args.action_exec_steps is None:
        args.action_exec_steps = min(DEFAULT_ACTION_EXEC_STEPS, args.action_horizon)
    if args.action_exec_steps < 1:
        parser.error("--action_exec_steps must be at least 1")
    if args.action_exec_start < 0:
        parser.error("--action_exec_start must be non-negative")
    if args.action_exec_start >= args.action_horizon:
        parser.error("--action_exec_start must be smaller than --action_horizon")
    if args.action_exec_start + args.action_exec_steps > args.action_horizon:
        parser.error("--action_exec_start + --action_exec_steps must be <= --action_horizon")
    if args.temporal_ensemble_decay < 0.0:
        parser.error("--temporal_ensemble_decay must be non-negative")
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        parser.error("--ema_decay must be in [0, 1); use 0 to disable EMA")
    if args.denoising_steps < 1:
        parser.error("--denoising_steps must be at least 1")
    if args.ddim_steps is not None and args.ddim_steps < 1:
        parser.error("--ddim_steps must be at least 1")
    if args.use_ddim and args.ddim_steps > args.denoising_steps:
        parser.error("--ddim_steps must be <= --denoising_steps")
    if args.min_sampling_std < 0.0:
        parser.error("--min_sampling_std must be non-negative")
    if args.min_logprob_std <= 0.0:
        parser.error("--min_logprob_std must be positive")
    if args.train_stride < 1:
        parser.error("--train_stride must be at least 1")
    if args.val_stride < 1:
        parser.error("--val_stride must be at least 1")
    needs_offline_data = (not args.eval_only) or args.eval_every_epochs > 0
    if needs_offline_data and args.stage_ratios is None:
        parser.error("--stage_ratios is required unless --eval_only True and --eval_every_epochs 0")
    if args.stage_sample_total < 0:
        parser.error("--stage_sample_total must be non-negative")
    if needs_offline_data and args.stage_sample_total < 1:
        parser.error("--stage_sample_total must be at least 1")
    if args.max_train_batches < 0:
        parser.error("--max_train_batches must be non-negative")
    if args.max_val_batches < 0:
        parser.error("--max_val_batches must be non-negative")
    if args.eval_only and args.resume_path is None:
        parser.error("--eval_only requires --resume_path")
    if args.n_eval_episodes < 1:
        parser.error("--n_eval_episodes must be at least 1")
    if args.eval_repeats < 1:
        parser.error("--eval_repeats must be at least 1")
    if args.eval_seed_base is not None and args.eval_seed_base < 0:
        parser.error("--eval_seed_base must be non-negative")
    if args.eval_env_max_steps < 1:
        parser.error("--eval_env_max_steps must be at least 1")
    if args.eval_reset_retry_attempts < 1:
        parser.error("--eval_reset_retry_attempts must be at least 1")
    if args.eval_success_distance_threshold < 0.0:
        parser.error("--eval_success_distance_threshold must be non-negative")
    if args.reset_bank_size_per_type < 0:
        parser.error("--reset_bank_size_per_type must be non-negative")
    if (
        args.eval_sequential_reset_bank_indices
        and args.reset_bank_size_per_type > 0
        and args.n_eval_episodes * args.eval_repeats > args.reset_bank_size_per_type
    ):
        parser.error(
            "--n_eval_episodes * --eval_repeats must be <= "
            "--reset_bank_size_per_type when --eval_sequential_reset_bank_indices True"
        )
    return args


def main():
    config = parse_args()
    if config.experiment_name == "bc_dppo_pretrain" and config.run_name:
        config.experiment_name = config.run_name
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_idx)
    if config.mujoco_gl:
        os.environ["MUJOCO_GL"] = config.mujoco_gl
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(config.gpu_idx)

    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generic_data_dirs = list(config.data_dirs)
    if generic_data_dirs == ["./data"] and (
        len(config.read_depth_data_dirs) > 0 or len(config.keyboard_data_dirs) > 0
    ):
        generic_data_dirs = []
    config.data_dirs = generic_data_dirs
    needs_offline_data = (not config.eval_only) or config.eval_every_epochs > 0

    print(f"[BC] device={device}")
    print(f"[BC] data_dirs={generic_data_dirs}")
    print(f"[BC] read_depth_data_dirs={config.read_depth_data_dirs}")
    print(f"[BC] keyboard_data_dirs={config.keyboard_data_dirs}")
    if config.stage_ratios is None:
        print("[BC] stage_ratios=None")
    else:
        print(f"[BC] stage_ratios={format_stage_ratios(config.stage_ratios)}")
    print(f"[BC] stage_sample_total={config.stage_sample_total}")
    print(f"[BC] eval_reset_types={config.eval_reset_types}")
    print(f"[BC] eval_warmup_steps={config.eval_warmup_steps}")
    print(f"[BC] eval_env_max_steps={config.eval_env_max_steps}")
    print(
        f"[BC] eval_success_distance_threshold={config.eval_success_distance_threshold}"
    )
    print(
        f"[BC] action_chunk horizon={config.action_horizon} "
        f"exec_start={config.action_exec_start} exec_steps={config.action_exec_steps}"
    )
    print(
        f"[BC] temporal_ensemble={config.temporal_ensemble} "
        f"decay={config.temporal_ensemble_decay}"
    )
    if config.progress_bar and tqdm is None:
        print("[BC] tqdm is not installed, progress bar disabled")
    swanlab_module = init_swanlab(config)

    action_dim = 7 if config.ur else 5
    train_dataset = None
    val_dataset = None
    train_loader = None
    val_loader = None

    if needs_offline_data:
        candidate_paths = discover_candidate_episode_paths(
            data_dirs=generic_data_dirs,
            read_depth_data_dirs=config.read_depth_data_dirs,
            keyboard_data_dirs=config.keyboard_data_dirs,
        )
        print(f"[BC] candidate episode files={len(candidate_paths)}")
        selected_candidate_paths, stage_selection_summary = select_episode_paths_by_stage_ratios(
            paths=candidate_paths,
            stage_ratios=config.stage_ratios,
            sample_total=config.stage_sample_total,
            seed=config.seed,
        )
        print(f"[BC] stage available counts={stage_selection_summary['available_counts']}")
        print(f"[BC] stage selected counts={stage_selection_summary['selected_counts']}")
        if stage_selection_summary["unknown_preview"]:
            print(
                "[BC] stage unknown files ignored preview="
                f"{stage_selection_summary['unknown_preview']}"
            )

        episode_specs, skipped = discover_episode_specs(
            data_dirs=generic_data_dirs,
            frame_stack=config.frame_stack,
            action_horizon=config.action_horizon,
            require_tactile=False,
            ur=config.ur,
            read_depth_data_dirs=config.read_depth_data_dirs,
            keyboard_data_dirs=config.keyboard_data_dirs,
            candidate_paths=selected_candidate_paths,
        )
        for path, reason in skipped[:20]:
            print(f"[BC] skipped {path}: {reason}")
        if len(skipped) > 20:
            print(f"[BC] ... {len(skipped) - 20} more skipped files")

        if not episode_specs:
            raise RuntimeError("no valid expert episodes found")

        tactile_episode_count = sum(spec.has_tactile for spec in episode_specs)
        if not config.vision_only_control and tactile_episode_count < len(episode_specs):
            if tactile_episode_count == 0:
                print("[BC] no tactile observations found in valid episodes, switching to vision_only_control=True")
                config.vision_only_control = True
            else:
                kept = [spec for spec in episode_specs if spec.has_tactile]
                skipped_missing = len(episode_specs) - len(kept)
                print(
                    f"[BC] skipping {skipped_missing} episodes without tactile because "
                    "vision_only_control=False"
                )
                episode_specs = kept

        if not episode_specs:
            raise RuntimeError(
                "no valid episodes left after applying --stage_ratios and validating selected files; "
                f"summary={stage_selection_summary}, skipped_preview={[(str(path), reason) for path, reason in skipped[:8]]}"
            )

        action_dims = sorted({spec.action_dim for spec in episode_specs})
        if len(action_dims) != 1:
            raise RuntimeError(f"inconsistent action dims across dataset: {action_dims}")
        action_dim = action_dims[0]
        expected_env_action_dim = 7 if config.ur else 5
        if action_dim != expected_env_action_dim:
            raise RuntimeError(
                "dataset action_dim does not match the environment control mode: "
                f"dataset={action_dim}, env_expected={expected_env_action_dim} (ur={config.ur}). "
                f"Use --ur {'True' if action_dim == 7 else 'False'} for this dataset, "
                "or switch to expert data collected with the matching action space."
            )

        train_specs, val_specs = split_episode_specs(episode_specs, config.val_ratio, config.seed)
        if not train_specs:
            raise RuntimeError("train split is empty after dataset split")

        train_dataset = ExpertSequenceDataset(
            episode_specs=train_specs,
            action_horizon=config.action_horizon,
            vision_only_control=config.vision_only_control,
            frame_stack=config.frame_stack,
            img_size=config.img_size,
            tactile_size=config.tactile_size,
            ur=config.ur,
            ee_pos_step=config.hdf5_ee_pos_step,
            ee_rot_step=config.hdf5_ee_rot_step,
            gripper_ctrl_step=config.hdf5_gripper_ctrl_step,
            hdf5_gripper_min=config.hdf5_gripper_min,
            hdf5_gripper_max=config.hdf5_gripper_max,
            sample_stride=config.train_stride,
        )
        val_dataset = (
            ExpertSequenceDataset(
                episode_specs=val_specs,
                action_horizon=config.action_horizon,
                vision_only_control=config.vision_only_control,
                frame_stack=config.frame_stack,
                img_size=config.img_size,
                tactile_size=config.tactile_size,
                ur=config.ur,
                ee_pos_step=config.hdf5_ee_pos_step,
                ee_rot_step=config.hdf5_ee_rot_step,
                gripper_ctrl_step=config.hdf5_gripper_ctrl_step,
                hdf5_gripper_min=config.hdf5_gripper_min,
                hdf5_gripper_max=config.hdf5_gripper_max,
                sample_stride=config.val_stride,
            )
            if val_specs
            else None
        )

        pin_memory = torch.cuda.is_available()
        train_loader_kwargs = {
            "dataset": train_dataset,
            "batch_size": config.batch_size,
            "shuffle": config.shuffle_train,
            "num_workers": config.num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if config.num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = config.prefetch_factor
            train_loader_kwargs["persistent_workers"] = True
        train_loader = DataLoader(**train_loader_kwargs)

        val_loader_kwargs = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader_kwargs = {
                "dataset": val_dataset,
                "batch_size": config.batch_size,
                "shuffle": False,
                "num_workers": config.num_workers,
                "pin_memory": pin_memory,
                "drop_last": False,
            }
            if config.num_workers > 0:
                val_loader_kwargs["prefetch_factor"] = config.prefetch_factor
                val_loader_kwargs["persistent_workers"] = True
        val_loader = (
            DataLoader(**val_loader_kwargs)
            if val_loader_kwargs is not None
            else None
        )

        npz_episode_count = sum(spec.file_format == "npz" for spec in episode_specs)
        hdf5_episode_count = sum(spec.file_format == "hdf5" for spec in episode_specs)
        print(
            f"[BC] valid episodes={len(episode_specs)} (npz={npz_episode_count}, hdf5={hdf5_episode_count}) "
            f"train={len(train_specs)} val={len(val_specs)} "
            f"train_samples={len(train_dataset)} val_samples={len(val_dataset) if val_dataset is not None else 0}"
        )
    else:
        print("[BC] eval-only env-only mode: skipping expert dataset discovery and stage sampling")

    mae = build_mae(config, device)
    if config.pretrained_mae_path:
        load_mae_checkpoint(
            mae,
            config.pretrained_mae_path,
            strict=config.pretrained_mae_strict,
        )

    policy = DPPOPolicy(
        mae_model=mae,
        dim_embeddings=config.dim_embedding,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        frame_stack=config.frame_stack,
        vision_only_control=config.vision_only_control,
        actor_hidden_dim=config.diffusion_hidden_dim,
        critic_hidden_dim=config.diffusion_hidden_dim,
        denoising_steps=config.denoising_steps,
        diffusion_time_dim=config.diffusion_time_dim,
        diffusion_denoiser_type=config.diffusion_denoiser_type,
        diffusion_transformer_depth=config.diffusion_transformer_depth,
        diffusion_transformer_heads=config.diffusion_transformer_heads,
        diffusion_transformer_dropout=config.diffusion_transformer_dropout,
        diffusion_cnn_depth=config.diffusion_cnn_depth,
        diffusion_cnn_kernel_size=config.diffusion_cnn_kernel_size,
        diffusion_cnn_dropout=config.diffusion_cnn_dropout,
        beta_schedule=config.diffusion_beta_schedule,
        beta_start=config.diffusion_beta_start,
        beta_end=config.diffusion_beta_end,
        use_ddim=config.use_ddim,
        ddim_steps=config.ddim_steps,
        denoised_clip_value=config.denoised_clip_value,
        randn_clip_value=config.randn_clip_value,
        final_action_clip_value=config.final_action_clip_value,
        eps_clip_value=config.eps_clip_value,
        min_sampling_std=config.min_sampling_std,
        min_logprob_std=config.min_logprob_std,
    ).to(device)

    if config.pretrained_policy_path:
        load_policy_checkpoint(policy, config.pretrained_policy_path, strict=False)

    if config.freeze_encoder:
        freeze_module(policy.encoder)
        policy.encoder.eval()
        print("[BC] encoder frozen")
    if config.freeze_critic:
        freeze_module(policy.critic)
        print("[BC] critic frozen")

    ema_policy = create_ema_policy(policy) if config.ema_decay > 0.0 else None
    if ema_policy is not None:
        print(f"[BC] EMA enabled (decay={config.ema_decay})")
    else:
        print("[BC] EMA disabled")

    trainable_params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_epoch = 1
    global_step = 0
    best_val_action_mse = math.inf
    best_eval_return = -math.inf
    best_eval_success_rate = -math.inf
    best_eval_min_success_distance = math.inf
    if config.resume_path:
        checkpoint = torch.load(config.resume_path, map_location="cpu")
        load_policy_checkpoint(policy, config.resume_path, strict=False)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema_policy is not None:
            load_ema_policy_checkpoint(ema_policy, checkpoint)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_action_mse = float(checkpoint.get("best_val_action_mse", best_val_action_mse))
        best_eval_return = float(checkpoint.get("best_eval_return", best_eval_return))
        best_eval_success_rate = float(
            checkpoint.get("best_eval_success_rate", best_eval_success_rate)
        )
        best_eval_min_success_distance = float(
            checkpoint.get("best_eval_min_success_distance", best_eval_min_success_distance)
        )
        print(f"[BC] resumed from {config.resume_path} at epoch {start_epoch}")

    eval_env = None
    if config.n_eval_episodes > 0 and config.env_eval_every_epochs > 0:
        eval_env = make_env(config, config.seed + 1000)

    eval_seed_base = None
    if eval_env is not None:
        total_eval_episodes = config.n_eval_episodes * max(config.eval_repeats, 1)
        eval_seed_base = resolve_eval_seed_base(config, total_eval_episodes)
        print(
            f"[BC] eval seeds: base={eval_seed_base} "
            f"episodes_per_repeat={config.n_eval_episodes} repeats={config.eval_repeats} "
            f"total={total_eval_episodes} reset_retry_attempts={config.eval_reset_retry_attempts}"
        )
        print(
            f"[BC] reset_bank_size_per_type={config.reset_bank_size_per_type} "
            f"reset_cache_dir={config.reset_cache_dir} "
            f"sequential_bank_indices={config.eval_sequential_reset_bank_indices}"
        )

    config_dict = vars(config).copy()
    config_dict["resolved_eval_seed_base"] = eval_seed_base
    eval_policy = ema_policy if ema_policy is not None else policy
    if config.eval_only:
        eval_epoch = max(start_epoch - 1, 0)
        log_record = {
            "epoch": eval_epoch,
            "global_step": global_step,
            "eval_only": True,
            "resume_path": config.resume_path,
        }
        if val_loader is not None and config.eval_every_epochs > 0:
            offline_metrics = evaluate_offline(
                policy=eval_policy,
                dataloader=val_loader,
                device=device,
                mae_loss_coef=config.mae_loss_coef,
                vision_only_control=config.vision_only_control,
                max_batches=config.max_val_batches,
                progress_bar=config.progress_bar,
                progress_desc="BC eval-only val",
            )
            if offline_metrics is not None:
                log_record.update({
                    "val/bc_loss": offline_metrics["bc_loss"],
                    "val/mae_loss": offline_metrics["mae_loss"],
                    "val/action_mse": offline_metrics["action_mse"],
                    "val/action_l1": offline_metrics["action_l1"],
                })
        if eval_env is not None:
            env_metrics = evaluate_in_env_repeats(
                policy=eval_policy,
                eval_env=eval_env,
                device=device,
                config=config,
                eval_seed_base=int(eval_seed_base),
                tag="eval_only",
                progress_desc="BC eval-only env",
            )
            log_record.update({
                "eval/mean_return": env_metrics["mean_return"],
                "eval/success_rate": env_metrics["success_rate"],
                "eval/success_count": env_metrics["success_count"],
                "eval/repeats": env_metrics["eval_repeats"],
                "eval/seed_base": env_metrics["eval_seed_base"],
                "eval/seed_start": env_metrics["seed_start"],
                "eval/seed_end": env_metrics["seed_end"],
                "eval/episode_seeds": env_metrics["episode_seeds"],
                "eval/repeat_seed_ranges": env_metrics["repeat_seed_ranges"],
                "eval/mean_length": env_metrics["mean_length"],
                "eval/min_length": env_metrics["min_length"],
                "eval/max_length": env_metrics["max_length"],
                "eval/warmup_steps": env_metrics["eval_warmup_steps"],
                "eval/reset_types": env_metrics["reset_types"],
                "eval/success_seeds": env_metrics["success_seeds"],
                "eval/success_gif_paths": env_metrics["success_gif_paths"],
                "eval/mean_min_success_distance": env_metrics["mean_min_success_distance"],
                "eval/mean_final_success_distance": env_metrics["mean_final_success_distance"],
                "eval/mean_min_dist_gripper_obj": env_metrics["mean_min_dist_gripper_obj"],
                "eval/mean_min_object_tilt_angle_rad": env_metrics["mean_min_object_tilt_angle_rad"],
                "eval/mean_final_object_tilt_angle_rad": env_metrics["mean_final_object_tilt_angle_rad"],
                "eval/mean_min_object_rotation_angle_rad": env_metrics["mean_min_object_rotation_angle_rad"],
                "eval/mean_final_object_rotation_angle_rad": env_metrics["mean_final_object_rotation_angle_rad"],
                "eval/mean_action_saturation_fraction": env_metrics["mean_action_saturation_fraction"],
                "eval/relaxed_success_rates": env_metrics["relaxed_success_rates"],
                "eval/reset_type_summary": env_metrics["reset_type_summary"],
                "eval/episode_records": env_metrics["episode_records"],
            })
        log_to_swanlab(swanlab_module, log_record, step=global_step)
        append_metrics(Path(config.log_path), log_record)
        summary_parts = [f"eval_only epoch={eval_epoch}"]
        if "val/action_mse" in log_record:
            summary_parts.append(f"val_mse={log_record['val/action_mse']:.6f}")
        if "eval/mean_return" in log_record:
            summary_parts.append(f"eval_return={log_record['eval/mean_return']:.4f}")
            summary_parts.append(f"eval_success={log_record['eval/success_rate']:.3f}")
            summary_parts.append(f"eval_len={log_record['eval/mean_length']:.1f}")
            summary_parts.append(
                f"eval_seeds={log_record['eval/seed_start']}-{log_record['eval/seed_end']}"
            )
            if log_record.get("eval/mean_min_success_distance") is not None:
                summary_parts.append(
                    f"eval_min_dist={log_record['eval/mean_min_success_distance']:.5f}"
                )
            if log_record.get("eval/mean_final_object_tilt_angle_rad") is not None:
                summary_parts.append(
                    f"eval_final_tilt={log_record['eval/mean_final_object_tilt_angle_rad']:.3f}"
                )
        print("[BC] " + " ".join(summary_parts))
        close_env(eval_env)
        if swanlab_module is not None and hasattr(swanlab_module, "finish"):
            swanlab_module.finish()
        return

    for epoch in range(start_epoch, config.epochs + 1):
        if config.freeze_encoder:
            policy.encoder.eval()

        train_metrics = train_one_epoch(
            policy=policy,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=config.max_grad_norm,
            mae_loss_coef=config.mae_loss_coef,
            vision_only_control=config.vision_only_control,
            ema_policy=ema_policy,
            ema_decay=config.ema_decay,
            max_batches=config.max_train_batches,
            progress_bar=config.progress_bar,
            progress_desc=f"BC train {epoch}/{config.epochs}",
        )
        global_step += len(train_dataset)

        log_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train/loss": train_metrics["loss"],
            "train/bc_loss": train_metrics["bc_loss"],
            "train/mae_loss": train_metrics["mae_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        should_eval_offline = (
            val_loader is not None
            and config.eval_every_epochs > 0
            and epoch % config.eval_every_epochs == 0
        )
        if should_eval_offline:
            offline_metrics = evaluate_offline(
                policy=eval_policy,
                dataloader=val_loader,
                device=device,
                mae_loss_coef=config.mae_loss_coef,
                vision_only_control=config.vision_only_control,
                max_batches=config.max_val_batches,
                progress_bar=config.progress_bar,
                progress_desc=f"BC val {epoch}/{config.epochs}",
            )
            if offline_metrics is not None:
                log_record.update({
                    "val/bc_loss": offline_metrics["bc_loss"],
                    "val/mae_loss": offline_metrics["mae_loss"],
                    "val/action_mse": offline_metrics["action_mse"],
                    "val/action_l1": offline_metrics["action_l1"],
                })
                if offline_metrics["action_mse"] < best_val_action_mse:
                    best_val_action_mse = offline_metrics["action_mse"]
                    save_checkpoint(
                        path=Path(config.best_val_path),
                        policy=policy,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        config=config_dict,
                        ema_policy=ema_policy,
                        extra={
                            "best_val_action_mse": best_val_action_mse,
                            "best_eval_return": best_eval_return,
                            "best_eval_success_rate": best_eval_success_rate,
                            "best_eval_min_success_distance": best_eval_min_success_distance,
                            "monitor": "val/action_mse",
                        },
                    )
                    print(
                        f"[BC] new best val checkpoint at epoch {epoch} "
                        f"(action_mse={best_val_action_mse:.6f})"
                    )

        should_eval_env = (
            eval_env is not None
            and config.env_eval_every_epochs > 0
            and epoch % config.env_eval_every_epochs == 0
        )
        if should_eval_env:
            env_metrics = evaluate_in_env_repeats(
                policy=eval_policy,
                eval_env=eval_env,
                device=device,
                config=config,
                eval_seed_base=int(eval_seed_base),
                tag=f"epoch_{epoch:04d}",
                progress_desc=f"BC env {epoch}/{config.epochs}",
            )
            log_record.update({
                "eval/mean_return": env_metrics["mean_return"],
                "eval/success_rate": env_metrics["success_rate"],
                "eval/success_count": env_metrics["success_count"],
                "eval/repeats": env_metrics["eval_repeats"],
                "eval/seed_base": env_metrics["eval_seed_base"],
                "eval/seed_start": env_metrics["seed_start"],
                "eval/seed_end": env_metrics["seed_end"],
                "eval/episode_seeds": env_metrics["episode_seeds"],
                "eval/repeat_seed_ranges": env_metrics["repeat_seed_ranges"],
                "eval/mean_length": env_metrics["mean_length"],
                "eval/min_length": env_metrics["min_length"],
                "eval/max_length": env_metrics["max_length"],
                "eval/warmup_steps": env_metrics["eval_warmup_steps"],
                "eval/reset_types": env_metrics["reset_types"],
                "eval/success_seeds": env_metrics["success_seeds"],
                "eval/success_gif_paths": env_metrics["success_gif_paths"],
                "eval/mean_min_success_distance": env_metrics["mean_min_success_distance"],
                "eval/mean_final_success_distance": env_metrics["mean_final_success_distance"],
                "eval/mean_min_dist_gripper_obj": env_metrics["mean_min_dist_gripper_obj"],
                "eval/mean_min_object_tilt_angle_rad": env_metrics["mean_min_object_tilt_angle_rad"],
                "eval/mean_final_object_tilt_angle_rad": env_metrics["mean_final_object_tilt_angle_rad"],
                "eval/mean_min_object_rotation_angle_rad": env_metrics["mean_min_object_rotation_angle_rad"],
                "eval/mean_final_object_rotation_angle_rad": env_metrics["mean_final_object_rotation_angle_rad"],
                "eval/mean_action_saturation_fraction": env_metrics["mean_action_saturation_fraction"],
                "eval/relaxed_success_rates": env_metrics["relaxed_success_rates"],
                "eval/reset_type_summary": env_metrics["reset_type_summary"],
                "eval/episode_records": env_metrics["episode_records"],
            })
            for eval_episode_idx, episode_length in enumerate(env_metrics["episode_lengths"]):
                log_record[f"eval/episode_{eval_episode_idx}_length"] = episode_length
            for eval_episode_idx, episode_record in enumerate(env_metrics["episode_records"]):
                log_record[f"eval/episode_{eval_episode_idx}_min_success_distance"] = (
                    episode_record["min_success_distance"]
                )
            if eval_is_better(
                env_metrics,
                best_eval_success_rate,
                best_eval_min_success_distance,
                best_eval_return,
            ):
                best_eval_return = float(env_metrics["mean_return"])
                best_eval_success_rate = float(env_metrics["success_rate"])
                mean_min_distance = env_metrics["mean_min_success_distance"]
                best_eval_min_success_distance = (
                    math.inf if mean_min_distance is None else float(mean_min_distance)
                )
                save_checkpoint(
                    path=Path(config.best_eval_path),
                    policy=policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    config=config_dict,
                    ema_policy=ema_policy,
                    extra={
                        "best_val_action_mse": best_val_action_mse,
                        "best_eval_return": best_eval_return,
                        "best_eval_success_rate": best_eval_success_rate,
                        "best_eval_min_success_distance": best_eval_min_success_distance,
                        "monitor": "eval/success_rate,min_success_distance,mean_return",
                    },
                )
                print(
                    f"[BC] new best env checkpoint at epoch {epoch} "
                    f"(success_rate={best_eval_success_rate:.3f}, "
                    f"mean_min_success_distance={best_eval_min_success_distance:.6f}, "
                    f"mean_return={best_eval_return:.4f})"
                )

        if config.save_every_epochs > 0 and epoch % config.save_every_epochs == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"bc_epoch_{epoch:04d}.pt"
            save_checkpoint(
                path=checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=config_dict,
                ema_policy=ema_policy,
                extra={
                    "best_val_action_mse": best_val_action_mse,
                    "best_eval_return": best_eval_return,
                    "best_eval_success_rate": best_eval_success_rate,
                    "best_eval_min_success_distance": best_eval_min_success_distance,
                },
            )
            print(f"[BC] checkpoint saved to {checkpoint_path}")

        log_to_swanlab(swanlab_module, log_record, step=global_step)
        append_metrics(Path(config.log_path), log_record)
        summary_parts = [
            f"epoch={epoch}",
            f"train_loss={train_metrics['loss']:.6f}",
            f"train_bc={train_metrics['bc_loss']:.6f}",
        ]
        if "val/action_mse" in log_record:
            summary_parts.append(f"val_mse={log_record['val/action_mse']:.6f}")
        if "eval/mean_return" in log_record:
            summary_parts.append(f"eval_return={log_record['eval/mean_return']:.4f}")
            summary_parts.append(f"eval_success={log_record['eval/success_rate']:.3f}")
            summary_parts.append(f"eval_len={log_record['eval/mean_length']:.1f}")
            summary_parts.append(
                f"eval_seeds={log_record['eval/seed_start']}-{log_record['eval/seed_end']}"
            )
            if log_record.get("eval/mean_min_success_distance") is not None:
                summary_parts.append(
                    f"eval_min_dist={log_record['eval/mean_min_success_distance']:.5f}"
                )
            if log_record.get("eval/mean_final_object_tilt_angle_rad") is not None:
                summary_parts.append(
                    f"eval_final_tilt={log_record['eval/mean_final_object_tilt_angle_rad']:.3f}"
                )
            if log_record.get("eval/mean_action_saturation_fraction") is not None:
                summary_parts.append(
                    f"eval_act_sat={log_record['eval/mean_action_saturation_fraction']:.3f}"
                )
        print("[BC] " + " ".join(summary_parts))

    save_checkpoint(
        path=Path(config.final_model_path),
        policy=policy,
        optimizer=optimizer,
        epoch=config.epochs,
        global_step=global_step,
        config=config_dict,
        ema_policy=ema_policy,
        extra={
            "best_val_action_mse": best_val_action_mse,
            "best_eval_return": best_eval_return,
            "best_eval_success_rate": best_eval_success_rate,
            "best_eval_min_success_distance": best_eval_min_success_distance,
        },
    )
    log_to_swanlab(
        swanlab_module,
        {
            "best/val_action_mse": best_val_action_mse,
            "best/eval_mean_return": best_eval_return,
            "best/eval_success_rate": best_eval_success_rate,
            "best/eval_min_success_distance": best_eval_min_success_distance,
        },
        step=global_step,
    )
    print(f"[BC] final model saved to {config.final_model_path}")

    close_env(eval_env)
    if swanlab_module is not None and hasattr(swanlab_module, "finish"):
        swanlab_module.finish()


if __name__ == "__main__":
    main()
