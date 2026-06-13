from __future__ import annotations

import argparse
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
    args, _ = parser.parse_known_args()

    target_gpu = str(args.gpu_idx)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", target_gpu)
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

from models.dppo_mae import prepare_vt_observations
from models.pretrain_models import VTT, VTMAE


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"boolean argument should be either True or False (got {value})")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def init_swanlab(config):
    if not config.use_swanlab:
        return None
    if swanlab is None:
        print("[MAE] swanlab is not installed, logging disabled")
        return None

    try:
        swanlab.init(
            project=config.project_name,
            experiment_name=config.experiment_name,
            config=vars(config),
            mode=config.swanlab_mode,
        )
        print(
            f"[MAE] swanlab enabled "
            f"(project={config.project_name}, experiment={config.experiment_name}, mode={config.swanlab_mode})"
        )
        return swanlab
    except Exception as exc:
        print(f"[MAE] swanlab init failed, logging disabled: {exc}")
        return None


def log_to_swanlab(swanlab_module, metrics: Dict[str, float], step: int) -> None:
    if swanlab_module is None:
        return
    swanlab_module.log(metrics, step=step)


@dataclass
class EpisodeSpec:
    path: Path
    file_format: str
    num_steps: int
    has_tactile: bool
    image_shape: Tuple[int, ...]
    tactile_shape: Optional[Tuple[int, ...]]
    pre_stacked: bool


class MAESequenceDataset(Dataset):
    def __init__(
        self,
        episode_specs: Sequence[EpisodeSpec],
        vision_only_control: bool,
        frame_stack: int,
        img_size: int,
        tactile_size: int,
    ) -> None:
        self.episode_specs = list(episode_specs)
        self.vision_only_control = bool(vision_only_control)
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.tactile_size = int(tactile_size)
        self.samples: List[Tuple[int, int]] = []
        for episode_idx, spec in enumerate(self.episode_specs):
            start_idx = 0 if spec.pre_stacked else max(self.frame_stack - 1, 0)
            for step_idx in range(start_idx, spec.num_steps):
                self.samples.append((episode_idx, step_idx))
        self._cache_episode_idx: Optional[int] = None
        self._cache_data: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        if self._cache_episode_idx == episode_idx and self._cache_data is not None:
            return self._cache_data

        spec = self.episode_specs[episode_idx]
        if spec.file_format == "npz":
            with np.load(spec.path, allow_pickle=True) as episode:
                payload = {
                    "images": np.asarray(episode["images"]),
                }
                if spec.has_tactile and not self.vision_only_control:
                    payload["tactiles"] = np.asarray(episode["tactiles"])
        elif spec.file_format == "hdf5":
            if h5py is None:
                raise ImportError("h5py is required to load hdf5 expert data")
            with h5py.File(spec.path, "r") as episode:
                if spec.pre_stacked:
                    payload = {
                        "images": np.asarray(episode["images"]),
                    }
                    if spec.has_tactile and not self.vision_only_control:
                        payload["tactiles"] = np.asarray(episode["tactiles"])
                else:
                    payload = {
                        "eye_in_hand": np.asarray(episode["eye_in_hand"]),
                        "eye_to_hand": np.asarray(episode["eye_to_hand"]),
                    }
                    if spec.has_tactile and not self.vision_only_control:
                        payload["tactile"] = np.asarray(episode["tactile"])
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
        if np.issubdtype(image.dtype, np.integer):
            resized = np.clip(np.rint(resized), 0, 255).astype(image.dtype)
        return resized

    def _resize_chw(self, image: np.ndarray, size: int) -> np.ndarray:
        if image.shape[-2] == size and image.shape[-1] == size:
            return image
        tensor = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        resized = tensor.squeeze(0).cpu().numpy()
        if np.issubdtype(image.dtype, np.integer):
            resized = np.clip(np.rint(resized), 0, 255).astype(image.dtype)
        return resized

    def _build_hdf5_observation(
        self,
        payload: Dict[str, np.ndarray],
        step_idx: int,
    ) -> Dict[str, torch.Tensor]:
        start_idx = step_idx - self.frame_stack + 1
        eye_in_hand = payload["eye_in_hand"][start_idx:step_idx + 1]
        eye_to_hand = payload["eye_to_hand"][start_idx:step_idx + 1]

        image_frames = []
        for frame_in_hand, frame_to_hand in zip(eye_in_hand, eye_to_hand):
            if frame_in_hand.shape[:2] != (self.img_size, self.img_size):
                frame_in_hand = self._resize_hwc(frame_in_hand, self.img_size)
            if frame_to_hand.shape[:2] != (self.img_size, self.img_size):
                frame_to_hand = self._resize_hwc(frame_to_hand, self.img_size)
            image_frames.append(np.concatenate((frame_in_hand, frame_to_hand), axis=-1))

        stacked_image = np.concatenate(image_frames, axis=-1)
        observation = {"image": torch.from_numpy(stacked_image)}

        if "tactile" in payload:
            tactile_frames = []
            for tactile in payload["tactile"][start_idx:step_idx + 1]:
                if tactile.shape[-2:] != (self.tactile_size, self.tactile_size):
                    tactile = self._resize_chw(tactile, self.tactile_size)
                tactile_frames.append(tactile)
            stacked_tactile = np.concatenate(tactile_frames, axis=0)
            observation["tactile"] = torch.from_numpy(stacked_tactile)

        return observation

    def __getitem__(self, index: int):
        episode_idx, step_idx = self.samples[index]
        payload = self._load_episode(episode_idx)

        spec = self.episode_specs[episode_idx]
        if spec.file_format == "hdf5" and not spec.pre_stacked:
            return self._build_hdf5_observation(payload, step_idx)

        observation = {"image": torch.from_numpy(payload["images"][step_idx])}
        if "tactiles" in payload:
            observation["tactile"] = torch.from_numpy(payload["tactiles"][step_idx])
        return observation


def discover_episode_specs(
    data_dirs: Sequence[str],
    frame_stack: int,
    require_tactile: bool,
) -> Tuple[List[EpisodeSpec], List[Tuple[Path, str]]]:
    episode_specs: List[EpisodeSpec] = []
    skipped: List[Tuple[Path, str]] = []
    expected_image_channels = 6 * frame_stack

    files: List[Path] = []
    for data_dir in data_dirs:
        root = Path(data_dir).expanduser()
        files.extend(sorted(root.glob("*.npz")))
        files.extend(sorted(root.glob("*.hdf5")))
        files.extend(sorted(root.glob("*.h5")))

    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".npz":
                with np.load(path, allow_pickle=True) as episode:
                    if "images" not in episode:
                        skipped.append((path, "missing images"))
                        continue

                    images = np.asarray(episode["images"])
                    if images.ndim != 4:
                        skipped.append((path, f"images ndim {images.ndim} != 4"))
                        continue
                    if images.shape[-1] != expected_image_channels:
                        skipped.append(
                            (path, f"image channels {images.shape[-1]} != expected {expected_image_channels}")
                        )
                        continue

                    has_tactile = "tactiles" in episode.files
                    tactile_shape = None
                    if has_tactile:
                        tactiles = np.asarray(episode["tactiles"])
                        tactile_shape = tuple(tactiles.shape)
                        if tactiles.shape[0] != images.shape[0]:
                            skipped.append((path, "tactiles/images length mismatch"))
                            continue
                    elif require_tactile:
                        skipped.append((path, "missing tactiles"))
                        continue

                    if images.shape[0] < 1:
                        skipped.append((path, "empty episode"))
                        continue

                    episode_specs.append(
                        EpisodeSpec(
                            path=path,
                            file_format="npz",
                            num_steps=int(images.shape[0]),
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
                    if "images" in episode:
                        images = episode["images"]
                        if len(images.shape) != 4:
                            skipped.append((path, f"images ndim {len(images.shape)} != 4"))
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
                            if tactiles.shape[0] != images.shape[0]:
                                skipped.append((path, "tactiles/images length mismatch"))
                                continue
                        elif require_tactile:
                            skipped.append((path, "missing tactiles"))
                            continue

                        if images.shape[0] < 1:
                            skipped.append((path, "empty episode"))
                            continue

                        episode_specs.append(
                            EpisodeSpec(
                                path=path,
                                file_format="hdf5",
                                num_steps=int(images.shape[0]),
                                has_tactile=has_tactile,
                                image_shape=tuple(images.shape),
                                tactile_shape=tactile_shape,
                                pre_stacked=True,
                            )
                        )
                        continue

                    if "eye_in_hand" not in episode or "eye_to_hand" not in episode:
                        skipped.append((path, "missing eye_in_hand / eye_to_hand"))
                        continue

                    eye_in_hand = episode["eye_in_hand"]
                    eye_to_hand = episode["eye_to_hand"]
                    if len(eye_in_hand.shape) != 4 or len(eye_to_hand.shape) != 4:
                        skipped.append((path, "camera datasets must be 4D"))
                        continue
                    if eye_in_hand.shape != eye_to_hand.shape:
                        skipped.append((path, "eye_in_hand / eye_to_hand shape mismatch"))
                        continue
                    if eye_in_hand.shape[-1] != 3 or eye_to_hand.shape[-1] != 3:
                        skipped.append((path, "camera channels must be RGB"))
                        continue
                    if eye_in_hand.shape[0] < frame_stack:
                        skipped.append((path, "too short for requested frame_stack"))
                        continue

                    has_tactile = "tactile" in episode.keys()
                    tactile_shape = None
                    if has_tactile:
                        tactile = episode["tactile"]
                        tactile_shape = tuple(tactile.shape)
                        if len(tactile.shape) != 4:
                            skipped.append((path, "tactile dataset must be 4D"))
                            continue
                        if tactile.shape[0] != eye_in_hand.shape[0]:
                            skipped.append((path, "tactile/camera length mismatch"))
                            continue
                        if tactile.shape[1] != 6:
                            skipped.append((path, f"expected tactile channel dim 6, got {tactile.shape[1]}"))
                            continue
                    elif require_tactile:
                        skipped.append((path, "missing tactile"))
                        continue

                    image_shape = (
                        int(eye_in_hand.shape[0]),
                        int(eye_in_hand.shape[1]),
                        int(eye_in_hand.shape[2]),
                        6,
                    )
                    episode_specs.append(
                        EpisodeSpec(
                            path=path,
                            file_format="hdf5",
                            num_steps=int(eye_in_hand.shape[0]),
                            has_tactile=has_tactile,
                            image_shape=image_shape,
                            tactile_shape=tactile_shape,
                            pre_stacked=False,
                        )
                    )
            else:
                skipped.append((path, f"unsupported suffix {suffix}"))
        except Exception as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))

    return episode_specs, skipped


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
        if key == "image" and raw_dtype == torch.uint8:
            value = value / 255.0
        if key == "tactile" and raw_dtype == torch.uint8:
            value = value / 255.0
        moved[key] = value
    return moved


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


def save_checkpoint(
    path: Path,
    mae_model: VTMAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Dict,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mae_state_dict": mae_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    mae_model: VTMAE,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("mae_state_dict", checkpoint.get("state_dict", checkpoint))
    mae_model.load_state_dict(state_dict, strict=True)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    print(f"[MAE] resumed from {path} (epoch={epoch}, global_step={global_step})")
    return epoch, global_step, best_val_loss


def mae_reconstruction_loss(
    mae_model: VTMAE,
    observations: Dict[str, torch.Tensor],
    frame_stack: int,
    device: torch.device,
    vision_only_control: bool,
) -> torch.Tensor:
    vt_obs = prepare_vt_observations(observations, frame_stack=frame_stack, device=device)
    return mae_model(
        vt_obs,
        use_vision=True,
        use_tactile=not vision_only_control,
    )


def train_one_epoch(
    mae_model: VTMAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config,
    progress_desc: str,
) -> Dict[str, float]:
    mae_model.train()
    total_loss = 0.0
    total_examples = 0

    progress = wrap_progress(
        dataloader,
        config.progress_bar,
        desc=progress_desc,
        unit="batch",
        leave=False,
    )
    try:
        for observations in progress:
            batch_obs = move_observations_to_device(observations, device)
            batch_size = batch_obs["image"].shape[0]
            loss = mae_reconstruction_loss(
                mae_model,
                batch_obs,
                frame_stack=config.frame_stack,
                device=device,
                vision_only_control=config.vision_only_control,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(mae_model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            set_progress_postfix(progress, loss=f"{loss.item():.4f}")
    finally:
        close_progress(progress)

    mean_divisor = max(total_examples, 1)
    return {"loss": total_loss / mean_divisor}


@torch.no_grad()
def evaluate(
    mae_model: VTMAE,
    dataloader: Optional[DataLoader],
    device: torch.device,
    config,
    progress_desc: str,
) -> Optional[Dict[str, float]]:
    if dataloader is None:
        return None

    mae_model.eval()
    total_loss = 0.0
    total_examples = 0

    progress = wrap_progress(
        dataloader,
        config.progress_bar,
        desc=progress_desc,
        unit="batch",
        leave=False,
    )
    try:
        for observations in progress:
            batch_obs = move_observations_to_device(observations, device)
            batch_size = batch_obs["image"].shape[0]
            loss = mae_reconstruction_loss(
                mae_model,
                batch_obs,
                frame_stack=config.frame_stack,
                device=device,
                vision_only_control=config.vision_only_control,
            )
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            set_progress_postfix(progress, loss=f"{loss.item():.4f}")
    finally:
        close_progress(progress)

    if total_examples == 0:
        return None
    return {"loss": total_loss / total_examples}


def append_metrics(log_path: Path, metrics: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser("Pretrain VTMAE on offline expert observations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_idx", type=int, default=int(TARGET_GPU))
    parser.add_argument("--data_dirs", type=str, nargs="+", default=["./data"])
    parser.add_argument("--run_name", type=str, default="mae_pretrain")
    parser.add_argument("--use_swanlab", type=str2bool, default=True)
    parser.add_argument("--project_name", type=str, default="DPPO-Robotics")
    parser.add_argument("--experiment_name", type=str, default="mae_pretrain")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")

    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--tactile_size", type=int, default=64)
    parser.add_argument("--vision_only_control", type=str2bool, default=False)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--progress_bar", type=str2bool, default=True)

    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--masking_ratio", type=float, default=0.75)

    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./logs/mae_checkpoints")
    parser.add_argument("--best_val_path", type=str, default="./logs/mae_checkpoints/best_val.pt")
    parser.add_argument("--final_model_path", type=str, default="./logs/mae_pretrained_final.pt")
    parser.add_argument("--log_path", type=str, default="./logs/mae_metrics.jsonl")
    return parser.parse_args()


def main():
    config = parse_args()
    if config.experiment_name == "mae_pretrain" and config.run_name:
        config.experiment_name = config.run_name

    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MAE] device={device}")
    print(f"[MAE] data_dirs={config.data_dirs}")
    if config.progress_bar and tqdm is None:
        print("[MAE] tqdm is not installed, progress bar disabled")
    swanlab_module = init_swanlab(config)

    episode_specs, skipped = discover_episode_specs(
        data_dirs=config.data_dirs,
        frame_stack=config.frame_stack,
        require_tactile=not config.vision_only_control,
    )
    for path, reason in skipped[:20]:
        print(f"[MAE] skipped {path}: {reason}")
    if len(skipped) > 20:
        print(f"[MAE] ... and {len(skipped) - 20} more skipped episodes")

    if not config.vision_only_control:
        tactile_episode_count = sum(int(spec.has_tactile) for spec in episode_specs)
        if tactile_episode_count == 0 and episode_specs:
            print("[MAE] no tactile observations found in valid episodes, switching to vision_only_control=True")
            config.vision_only_control = True
        elif tactile_episode_count < len(episode_specs):
            raise RuntimeError(
                "some episodes are missing tactile observations while vision_only_control=False"
            )

    if not episode_specs:
        raise RuntimeError("no valid episodes found for MAE pretraining")

    train_specs, val_specs = split_episode_specs(
        episode_specs=episode_specs,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    if not train_specs:
        raise RuntimeError("training split is empty after dataset split")

    print(
        f"[MAE] episodes: total={len(episode_specs)} train={len(train_specs)} val={len(val_specs)} "
        f"vision_only={config.vision_only_control}"
    )

    train_dataset = MAESequenceDataset(
        episode_specs=train_specs,
        vision_only_control=config.vision_only_control,
        frame_stack=config.frame_stack,
        img_size=config.img_size,
        tactile_size=config.tactile_size,
    )
    val_dataset = (
        MAESequenceDataset(
            episode_specs=val_specs,
            vision_only_control=config.vision_only_control,
            frame_stack=config.frame_stack,
            img_size=config.img_size,
            tactile_size=config.tactile_size,
        )
        if val_specs
        else None
    )

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if config.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["persistent_workers"] = True
        if config.prefetch_factor is not None and config.prefetch_factor > 0:
            train_loader_kwargs["prefetch_factor"] = config.prefetch_factor
            val_loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs) if val_dataset is not None else None

    mae_model = build_mae(config, device)
    optimizer = torch.optim.AdamW(
        mae_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.log_path)

    start_epoch = 1
    global_step = 0
    best_val_loss = math.inf
    if config.resume_path:
        resume_epoch, global_step, best_val_loss = load_checkpoint(
            config.resume_path,
            mae_model,
            optimizer,
        )
        start_epoch = resume_epoch + 1

    config_dict = vars(config).copy()
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = train_one_epoch(
            mae_model=mae_model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            progress_desc=f"MAE train {epoch}/{config.epochs}",
        )
        global_step += len(train_dataset)

        log_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train/recon_loss": train_metrics["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        should_eval = (
            val_loader is not None
            and config.eval_every_epochs > 0
            and epoch % config.eval_every_epochs == 0
        )
        if should_eval:
            val_metrics = evaluate(
                mae_model=mae_model,
                dataloader=val_loader,
                device=device,
                config=config,
                progress_desc=f"MAE val {epoch}/{config.epochs}",
            )
            if val_metrics is not None:
                log_record["val/recon_loss"] = val_metrics["loss"]
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(
                        path=Path(config.best_val_path),
                        mae_model=mae_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        config=config_dict,
                        best_val_loss=best_val_loss,
                    )
                    print(
                        f"[MAE] new best val checkpoint at epoch {epoch} "
                        f"(recon_loss={best_val_loss:.6f})"
                    )

        append_metrics(log_path, log_record)
        log_to_swanlab(swanlab_module, log_record, step=global_step)

        summary = (
            f"[MAE] epoch={epoch}/{config.epochs} "
            f"train_recon={train_metrics['loss']:.6f}"
        )
        if "val/recon_loss" in log_record:
            summary += f" val_recon={log_record['val/recon_loss']:.6f}"
        print(summary)

        if config.save_every_epochs > 0 and epoch % config.save_every_epochs == 0:
            save_checkpoint(
                path=checkpoint_dir / f"mae_epoch_{epoch:04d}.pt",
                mae_model=mae_model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=config_dict,
                best_val_loss=best_val_loss,
            )

    save_checkpoint(
        path=Path(config.final_model_path),
        mae_model=mae_model,
        optimizer=optimizer,
        epoch=config.epochs,
        global_step=global_step,
        config=config_dict,
        best_val_loss=best_val_loss,
    )
    print(f"Final MAE checkpoint saved to {config.final_model_path}")


if __name__ == "__main__":
    main()
