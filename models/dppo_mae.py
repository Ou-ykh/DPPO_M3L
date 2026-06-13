import copy
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.pretrain_models import VTT


def stack_observations(observations: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        key: np.stack([obs[key] for obs in observations], axis=0).astype(np.float32)
        for key in observations[0].keys()
    }


def copy_observation(observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: value.copy() for key, value in observation.items()}


def observation_to_eval_frame(observation: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    image = observation.get("image")
    if image is None:
        return None

    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return None

    channels = image.shape[-1]
    if channels >= 6:
        current = image[..., -6:]
        frame = np.concatenate((current[..., :3], current[..., 3:6]), axis=1)
    elif channels >= 3:
        frame = image[..., -3:]
    else:
        frame = np.repeat(image[..., :1], 3, axis=-1)

    frame = np.nan_to_num(frame)
    if frame.dtype != np.uint8:
        if float(np.max(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def save_gif(frames: List[np.ndarray], path: str, fps: int = 20) -> bool:
    if not frames:
        return False

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fps = max(int(fps), 1)
    try:
        import imageio

        imageio.mimsave(path, frames, fps=fps)
        return True
    except Exception as imageio_error:
        try:
            from PIL import Image

            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(int(1000 / fps), 1),
                loop=0,
            )
            return True
        except Exception as pil_error:
            print(
                "[DPPO] failed to save eval GIF. "
                f"imageio error: {imageio_error}; PIL error: {pil_error}"
            )
            return False


def avoid_existing_path(path: str) -> str:
    if not os.path.exists(path):
        return path

    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    stem, suffix = os.path.splitext(filename)
    for index in range(1, 10000):
        candidate = os.path.join(directory, f"{stem}_dup{index}{suffix}")
        if not os.path.exists(candidate):
            print(f"[DPPO] checkpoint path exists, saving to {candidate} instead")
            return candidate
    raise RuntimeError(f"unable to find a free checkpoint path near {path}")


def prepare_vt_observations(
    observations: Dict[str, torch.Tensor],
    frame_stack: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    vt_observations = {}

    if "image" in observations:
        image = observations["image"]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image, dtype=torch.float32, device=device)
        else:
            image = image.to(device=device, dtype=torch.float32)

        if image.dim() == 5:
            image = image.permute(0, 2, 3, 1, 4)
            image = image.reshape(image.shape[0], image.shape[1], image.shape[2], -1)

        vt_observations["image"] = image.permute(0, 3, 1, 2)

    if "tactile" in observations:
        tactile = observations["tactile"]
        if not torch.is_tensor(tactile):
            tactile = torch.as_tensor(tactile, dtype=torch.float32, device=device)
        else:
            tactile = tactile.to(device=device, dtype=torch.float32)

        if tactile.dim() == 5:
            tactile = tactile.reshape(tactile.shape[0], -1, tactile.shape[3], tactile.shape[4])

        n_tactiles = tactile.shape[1] // frame_stack
        n_sensors = n_tactiles // 3
        idx = []
        for frame_idx in range(frame_stack):
            idx.extend([
                frame_idx * n_tactiles + 0,
                frame_idx * n_tactiles + 1,
                frame_idx * n_tactiles + 2,
            ])
        idx = torch.as_tensor(idx, dtype=torch.long, device=device)

        for tactile_idx in range(n_sensors):
            tactile_key = f"tactile{tactile_idx + 1}"
            tactile_sensor = tactile[:, idx + 3 * tactile_idx]
            if float(tactile_sensor.min().item()) < -0.05:
                tactile_sensor = (tactile_sensor + 1.0) / 2.0
            elif float(tactile_sensor.max().item()) > 1.5:
                tactile_sensor = tactile_sensor / 255.0
            vt_observations[tactile_key] = tactile_sensor.clamp(0.0, 1.0)

    return vt_observations


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        embeddings = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if embeddings.shape[-1] < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - embeddings.shape[-1]))

        return embeddings


class MAEVitEncoder(nn.Module):
    def __init__(
        self,
        mae_model: nn.Module,
        dim_embeddings: int,
        frame_stack: int,
        vision_only_control: bool = False,
    ):
        super().__init__()
        self.mae_model = mae_model
        self.frame_stack = frame_stack
        self.vision_only_control = vision_only_control

        self.vit_layer = VTT(
            image_size=(64, 64),
            tactile_size=(32, 32),
            image_patch_size=8,
            tactile_patch_size=4,
            dim=dim_embeddings,
            depth=1,
            heads=4,
            mlp_dim=dim_embeddings * 2,
            num_tactiles=2,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        vt_observations = prepare_vt_observations(observations, self.frame_stack, device)
        use_tactile = (not self.vision_only_control) and any(
            key.startswith("tactile") for key in vt_observations
        )
        embeddings = self.mae_model.get_embeddings(
            vt_observations,
            eval=not self.training,
            use_tactile=use_tactile,
        )
        embeddings = self.vit_layer.transformer(embeddings)
        return torch.mean(embeddings, dim=1)

    def reconstruction_loss(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        vt_observations = prepare_vt_observations(observations, self.frame_stack, device)
        if self.vision_only_control or not any(key.startswith("tactile") for key in vt_observations):
            vt_observations = {"image": vt_observations["image"]}
        return self.mae_model(vt_observations)


class DenoisingMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        time_dim: int,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        return self.net(torch.cat((latent, noisy_action, time_embedding), dim=-1))


class DenoisingTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        time_dim: int,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by heads for Transformer denoiser "
                f"({hidden_dim} % {heads} != 0)"
            )

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_chunk_dim = action_dim * action_horizon
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, action_horizon + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_action.shape[0]
        action_tokens = noisy_action.reshape(batch_size, self.action_horizon, self.action_dim)
        action_tokens = self.action_proj(action_tokens)

        time_embedding = self.time_mlp(timesteps)
        condition_token = self.condition_proj(
            torch.cat((latent, time_embedding), dim=-1)
        ).unsqueeze(1)

        tokens = torch.cat((condition_token, action_tokens), dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1]]
        tokens = self.transformer(tokens)
        action_tokens = self.output_norm(tokens[:, 1:])
        predicted_noise = self.output_proj(action_tokens)
        return predicted_noise.reshape(batch_size, self.action_chunk_dim)


class FiLMTemporalConvBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"cnn kernel_size must be odd, got {kernel_size}")

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(x + h)


class DenoisingCNN(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        time_dim: int,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"cnn depth must be at least 1, got {depth}")

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_chunk_dim = action_dim * action_horizon
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                FiLMTemporalConvBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_action.shape[0]
        action_tokens = noisy_action.reshape(batch_size, self.action_horizon, self.action_dim)
        action_tokens = self.action_proj(action_tokens) + self.pos_embedding

        time_embedding = self.time_mlp(timesteps)
        cond = self.condition_proj(torch.cat((latent, time_embedding), dim=-1))

        x = action_tokens.transpose(1, 2)
        for block in self.blocks:
            x = block(x, cond)

        action_tokens = x.transpose(1, 2)
        predicted_noise = self.output_proj(self.output_norm(action_tokens))
        return predicted_noise.reshape(batch_size, self.action_chunk_dim)


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0.0, a_max=0.999)
    return torch.tensor(betas, dtype=dtype)


class GaussianDiffusionActor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int = 1,
        hidden_dim: int = 256,
        time_dim: int = 64,
        denoiser_type: str = "transformer",
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        cnn_depth: int = 4,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.0,
        denoising_steps: int = 100,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        use_ddim: bool = True,
        ddim_steps: Optional[int] = 5,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = 1.0,
        eps_clip_value: Optional[float] = None,
        min_sampling_std: float = 0.04,
        min_logprob_std: float = 0.04,
    ):
        super().__init__()
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be at least 1")
        if action_horizon < 1:
            raise ValueError("action_horizon must be at least 1")

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_chunk_dim = action_dim * action_horizon
        self.training_denoising_steps = int(denoising_steps)
        self.use_ddim = bool(use_ddim)
        self.ddim_steps = int(ddim_steps or denoising_steps)
        self.ddim_steps = max(1, min(self.ddim_steps, self.training_denoising_steps))
        self.sampling_steps = self.ddim_steps if self.use_ddim else self.training_denoising_steps
        self.denoising_steps = self.sampling_steps
        self.beta_schedule = beta_schedule.lower()
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = float(randn_clip_value)
        self.final_action_clip_value = final_action_clip_value
        self.eps_clip_value = eps_clip_value
        self.min_sampling_std = float(min_sampling_std)
        self.min_logprob_std = float(min_logprob_std)
        self.denoiser_type = denoiser_type.lower()
        if self.denoiser_type == "mlp":
            self.denoiser = DenoisingMLP(latent_dim, self.action_chunk_dim, hidden_dim, time_dim)
        elif self.denoiser_type == "transformer":
            self.denoiser = DenoisingTransformer(
                latent_dim=latent_dim,
                action_dim=action_dim,
                action_horizon=action_horizon,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                depth=transformer_depth,
                heads=transformer_heads,
                dropout=transformer_dropout,
            )
        elif self.denoiser_type == "cnn":
            self.denoiser = DenoisingCNN(
                latent_dim=latent_dim,
                action_dim=action_dim,
                action_horizon=action_horizon,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                depth=cnn_depth,
                kernel_size=cnn_kernel_size,
                dropout=cnn_dropout,
            )
        else:
            raise ValueError(
                f"unknown denoiser_type '{denoiser_type}', expected 'transformer', 'cnn', or 'mlp'"
            )

        if self.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.training_denoising_steps)
        elif self.beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.training_denoising_steps)
        else:
            raise ValueError("beta_schedule must be either 'cosine' or 'linear'")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]), dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))
        self.register_buffer("posterior_log_variance", posterior_variance.clamp_min(1e-20).log())
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @staticmethod
    def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return buffer.gather(0, timesteps).reshape(-1, *([1] * (x.dim() - 1)))

    def _sampling_timesteps(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_ddim:
            if self.sampling_steps == 1:
                timesteps = torch.tensor([self.training_denoising_steps - 1], device=device, dtype=torch.long)
            else:
                timesteps = torch.linspace(
                    self.training_denoising_steps - 1,
                    0,
                    self.sampling_steps,
                    device=device,
                ).round().long()
        else:
            timesteps = torch.arange(
                self.training_denoising_steps - 1,
                -1,
                -1,
                device=device,
                dtype=torch.long,
            )
        prev_timesteps = torch.cat((timesteps[1:], torch.full((1,), -1, device=device, dtype=torch.long)))
        return timesteps, prev_timesteps

    def _predict_noise(self, latent: torch.Tensor, noisy_action: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.denoiser(latent, noisy_action, timesteps)
        if self.eps_clip_value is not None:
            predicted_noise = torch.clamp(predicted_noise, -self.eps_clip_value, self.eps_clip_value)
        return predicted_noise

    def _predict_x0(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if predicted_noise is None:
            predicted_noise = self._predict_noise(latent, noisy_action, timesteps)
        sqrt_recip_alpha_bar_t = self._extract(
            self.sqrt_recip_alphas_cumprod,
            timesteps,
            noisy_action,
        )
        sqrt_recipm1_alpha_bar_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod,
            timesteps,
            noisy_action,
        )
        x0 = sqrt_recip_alpha_bar_t * noisy_action - sqrt_recipm1_alpha_bar_t * predicted_noise
        if self.denoised_clip_value is not None:
            x0 = torch.clamp(x0, -self.denoised_clip_value, self.denoised_clip_value)
            sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_action)
            sqrt_one_minus_alpha_bar_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                timesteps,
                noisy_action,
            )
            predicted_noise = (noisy_action - sqrt_alpha_bar_t * x0) / sqrt_one_minus_alpha_bar_t.clamp_min(1e-12)
        return x0, predicted_noise

    def _ddpm_parameters(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, _ = self._predict_x0(latent, noisy_action, timesteps)
        mean = (
            self._extract(self.posterior_mean_coef1, timesteps, noisy_action) * x0
            + self._extract(self.posterior_mean_coef2, timesteps, noisy_action) * noisy_action
        )
        std = torch.sqrt(self._extract(self.posterior_variance, timesteps, noisy_action)).expand_as(mean)
        zero_std = torch.zeros_like(std)
        if deterministic:
            std = torch.where(timesteps.reshape(-1, *([1] * (std.dim() - 1))) == 0, zero_std, std.clamp_min(1e-3))
        else:
            std = std.clamp_min(self.min_sampling_std)
        return mean, std

    def _ddim_parameters(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        prev_timesteps: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_noise = self._predict_noise(latent, noisy_action, timesteps)
        x0, predicted_noise = self._predict_x0(latent, noisy_action, timesteps, predicted_noise)
        alpha_prev = torch.ones(
            timesteps.shape[0],
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        )
        has_prev = prev_timesteps >= 0
        if has_prev.any():
            alpha_prev[has_prev] = self.alphas_cumprod[prev_timesteps[has_prev]].to(noisy_action.dtype)
        alpha_prev = alpha_prev.reshape(-1, *([1] * (noisy_action.dim() - 1)))
        mean = torch.sqrt(alpha_prev) * x0 + torch.sqrt((1.0 - alpha_prev).clamp_min(0.0)) * predicted_noise
        if deterministic:
            std = torch.zeros_like(mean)
        else:
            std = torch.full_like(mean, self.min_sampling_std)
        return mean, std

    def transition_parameters(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool = False,
        prev_timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_ddim:
            if prev_timesteps is None:
                prev_timesteps = torch.full_like(timesteps, -1)
            return self._ddim_parameters(latent, noisy_action, timesteps, prev_timesteps, deterministic)
        return self._ddpm_parameters(latent, noisy_action, timesteps, deterministic)

    def transition_distribution(
        self,
        latent: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        for_logprob: bool,
        prev_timesteps: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Normal:
        mean, std = self.transition_parameters(
            latent,
            noisy_action,
            timesteps,
            deterministic=False,
            prev_timesteps=prev_timesteps,
        )
        std_floor = self.min_logprob_std if for_logprob else self.min_sampling_std
        return torch.distributions.Normal(mean, std.clamp_min(std_floor))

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_start) * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                timesteps,
                x_start,
            ) * noise
        )

    def sample(
        self,
        latent: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = latent.shape[0]
        action = torch.randn(batch_size, self.action_chunk_dim, device=latent.device)
        action = action.clamp(-self.randn_clip_value, self.randn_clip_value)
        chains = [action]
        log_probs = []

        sample_timesteps, prev_sample_timesteps = self._sampling_timesteps(latent.device)
        for reverse_step, (timestep_value, prev_timestep_value) in enumerate(
            zip(sample_timesteps.tolist(), prev_sample_timesteps.tolist())
        ):
            timesteps = torch.full(
                (batch_size,),
                timestep_value,
                device=latent.device,
                dtype=torch.long,
            )
            prev_timesteps = torch.full(
                (batch_size,),
                prev_timestep_value,
                device=latent.device,
                dtype=torch.long,
            )
            mean, std = self.transition_parameters(
                latent,
                action,
                timesteps,
                deterministic=deterministic,
                prev_timesteps=prev_timesteps,
            )
            if deterministic and self.use_ddim:
                next_action = mean
            else:
                noise = torch.randn_like(action).clamp(-self.randn_clip_value, self.randn_clip_value)
                next_action = mean + std * noise
            if self.final_action_clip_value is not None and reverse_step == len(sample_timesteps) - 1:
                next_action = torch.clamp(next_action, -self.final_action_clip_value, self.final_action_clip_value)
            log_std = std.clamp_min(self.min_logprob_std)
            log_probs.append(torch.distributions.Normal(mean, log_std).log_prob(next_action).sum(dim=-1))
            chains.append(next_action)
            action = next_action

        if self.final_action_clip_value is not None:
            action = torch.clamp(action, -self.final_action_clip_value, self.final_action_clip_value)
        action = action.reshape(batch_size, self.action_horizon, self.action_dim)
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return action, chains, log_probs

    def get_log_probs(
        self,
        latent: torch.Tensor,
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        denoising_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_timesteps, prev_sample_timesteps = self._sampling_timesteps(latent.device)
        denoising_indices = denoising_indices.clamp(0, self.sampling_steps - 1)
        timesteps = sample_timesteps[denoising_indices]
        prev_timesteps = prev_sample_timesteps[denoising_indices]
        distribution = self.transition_distribution(
            latent,
            chains_prev,
            timesteps,
            for_logprob=True,
            prev_timesteps=prev_timesteps,
        )
        log_probs = distribution.log_prob(chains_next).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy


class DPPOPolicy(nn.Module):
    def __init__(
        self,
        mae_model: nn.Module,
        dim_embeddings: int,
        action_dim: int,
        action_horizon: int,
        frame_stack: int,
        vision_only_control: bool = False,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        denoising_steps: int = 8,
        diffusion_time_dim: int = 64,
        diffusion_denoiser_type: str = "transformer",
        diffusion_transformer_depth: int = 2,
        diffusion_transformer_heads: int = 4,
        diffusion_transformer_dropout: float = 0.0,
        diffusion_cnn_depth: int = 4,
        diffusion_cnn_kernel_size: int = 3,
        diffusion_cnn_dropout: float = 0.0,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        use_ddim: bool = True,
        ddim_steps: Optional[int] = 5,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = 1.0,
        eps_clip_value: Optional[float] = None,
        min_sampling_std: float = 0.04,
        min_logprob_std: float = 0.04,
    ):
        super().__init__()
        self.encoder = MAEVitEncoder(
            mae_model=mae_model,
            dim_embeddings=dim_embeddings,
            frame_stack=frame_stack,
            vision_only_control=vision_only_control,
        )
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_chunk_dim = action_dim * action_horizon
        self.actor_body = nn.Sequential(
            nn.Linear(dim_embeddings, actor_hidden_dim),
            nn.SiLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.SiLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(dim_embeddings, critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self.actor = GaussianDiffusionActor(
            latent_dim=actor_hidden_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=actor_hidden_dim,
            time_dim=diffusion_time_dim,
            denoiser_type=diffusion_denoiser_type,
            transformer_depth=diffusion_transformer_depth,
            transformer_heads=diffusion_transformer_heads,
            transformer_dropout=diffusion_transformer_dropout,
            cnn_depth=diffusion_cnn_depth,
            cnn_kernel_size=diffusion_cnn_kernel_size,
            cnn_dropout=diffusion_cnn_dropout,
            denoising_steps=denoising_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            denoised_clip_value=denoised_clip_value,
            randn_clip_value=randn_clip_value,
            final_action_clip_value=final_action_clip_value,
            eps_clip_value=eps_clip_value,
            min_sampling_std=min_sampling_std,
            min_logprob_std=min_logprob_std,
        )

    @property
    def denoising_steps(self) -> int:
        return self.actor.denoising_steps

    def encode(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(observations)

    def actor_latent(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor_body(self.encode(observations))

    def value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self.encode(observations)).squeeze(-1)

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(observations)
        actor_latent = self.actor_body(features)
        values = self.critic(features).squeeze(-1)
        actions, chains, log_probs = self.actor.sample(actor_latent, deterministic=deterministic)
        return actions, chains, log_probs, values

    def evaluate_denoising_step(
        self,
        observations: Dict[str, torch.Tensor],
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        denoising_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(observations)
        actor_latent = self.actor_body(features)
        values = self.critic(features).squeeze(-1)
        log_probs, entropy = self.actor.get_log_probs(
            actor_latent,
            chains_prev,
            chains_next,
            denoising_indices,
        )
        return values, log_probs, entropy


class StateMLPEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        dim_embeddings: int,
        hidden_dim: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"state encoder depth must be >= 1, got {depth}")

        layers = []
        in_dim = state_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim_embeddings))
        layers.append(nn.LayerNorm(dim_embeddings))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = observations["state"]
        return self.net(state)

    def reconstruction_loss(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = observations["state"]
        return torch.zeros((), dtype=state.dtype, device=state.device)


class DPPOStatePolicy(nn.Module):
    """State-observation DPPO policy for high-throughput MJWarp rollouts.

    This keeps the diffusion actor and critic interface used by DPPOTrainer, but
    replaces the MAE/VIT visual-tactile encoder with a lightweight state MLP.
    """

    def __init__(
        self,
        state_dim: int,
        dim_embeddings: int,
        action_dim: int,
        action_horizon: int = 1,
        state_encoder_hidden_dim: int = 256,
        state_encoder_depth: int = 2,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        denoising_steps: int = 8,
        diffusion_time_dim: int = 64,
        diffusion_denoiser_type: str = "transformer",
        diffusion_transformer_depth: int = 2,
        diffusion_transformer_heads: int = 4,
        diffusion_transformer_dropout: float = 0.0,
        diffusion_cnn_depth: int = 4,
        diffusion_cnn_kernel_size: int = 3,
        diffusion_cnn_dropout: float = 0.0,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        use_ddim: bool = True,
        ddim_steps: Optional[int] = 5,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = 1.0,
        eps_clip_value: Optional[float] = None,
        min_sampling_std: float = 0.04,
        min_logprob_std: float = 0.04,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_chunk_dim = action_dim * action_horizon
        self.encoder = StateMLPEncoder(
            state_dim=state_dim,
            dim_embeddings=dim_embeddings,
            hidden_dim=state_encoder_hidden_dim,
            depth=state_encoder_depth,
        )
        self.actor_body = nn.Sequential(
            nn.Linear(dim_embeddings, actor_hidden_dim),
            nn.SiLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.SiLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(dim_embeddings, critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self.actor = GaussianDiffusionActor(
            latent_dim=actor_hidden_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=actor_hidden_dim,
            time_dim=diffusion_time_dim,
            denoiser_type=diffusion_denoiser_type,
            transformer_depth=diffusion_transformer_depth,
            transformer_heads=diffusion_transformer_heads,
            transformer_dropout=diffusion_transformer_dropout,
            cnn_depth=diffusion_cnn_depth,
            cnn_kernel_size=diffusion_cnn_kernel_size,
            cnn_dropout=diffusion_cnn_dropout,
            denoising_steps=denoising_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            denoised_clip_value=denoised_clip_value,
            randn_clip_value=randn_clip_value,
            final_action_clip_value=final_action_clip_value,
            eps_clip_value=eps_clip_value,
            min_sampling_std=min_sampling_std,
            min_logprob_std=min_logprob_std,
        )

    @property
    def denoising_steps(self) -> int:
        return self.actor.denoising_steps

    def encode(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(observations)

    def actor_latent(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor_body(self.encode(observations))

    def value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self.encode(observations)).squeeze(-1)

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(observations)
        actor_latent = self.actor_body(features)
        values = self.critic(features).squeeze(-1)
        actions, chains, log_probs = self.actor.sample(actor_latent, deterministic=deterministic)
        return actions, chains, log_probs, values

    def evaluate_denoising_step(
        self,
        observations: Dict[str, torch.Tensor],
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        denoising_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(observations)
        actor_latent = self.actor_body(features)
        values = self.critic(features).squeeze(-1)
        log_probs, entropy = self.actor.get_log_probs(
            actor_latent,
            chains_prev,
            chains_next,
            denoising_indices,
        )
        return values, log_probs, entropy


@dataclass
class DPPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    mae_loss: float


class RewardNormalizer:
    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        batch_mean = float(np.mean(rewards))
        batch_var = float(np.var(rewards))
        batch_count = rewards.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

        return (rewards - self.mean) / (np.sqrt(self.var) + 1e-8)


class DPPOTrainer:
    def __init__(
        self,
        policy: DPPOPolicy,
        envs,
        device: torch.device,
        n_steps: int,
        batch_size: int,
        update_epochs: int,
        learning_rate: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        mae_loss_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        norm_reward: bool = False,
        action_exec_index: int = 0,
        action_exec_steps: int = 1,
    ):
        self.policy = policy.to(device)
        self.envs = envs
        self.device = device
        self.n_envs = len(envs)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.mae_loss_coef = mae_loss_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.reward_normalizer = RewardNormalizer() if norm_reward else None
        if action_exec_steps < 1:
            raise ValueError(f"action_exec_steps must be at least 1, got {action_exec_steps}")
        if action_exec_index < 0 or action_exec_index >= self.policy.action_horizon:
            raise ValueError(
                f"action_exec_index must be in [0, {self.policy.action_horizon - 1}], "
                f"got {action_exec_index}"
            )
        if action_exec_index + action_exec_steps > self.policy.action_horizon:
            raise ValueError(
                "action_exec_index + action_exec_steps must be <= action_horizon "
                f"({action_exec_index} + {action_exec_steps} > {self.policy.action_horizon})"
            )
        self.action_exec_index = action_exec_index
        self.action_exec_steps = action_exec_steps
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate)

        self.current_obs = None
        self.num_timesteps = 0
        self.iteration = 0

    def _obs_to_torch(self, observations: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in observations.items()
        }

    def reset_envs(self) -> Dict[str, np.ndarray]:
        observations = []
        for env_idx, env in enumerate(self.envs):
            obs, _ = env.reset(seed=env_idx)
            observations.append(obs)
        self.current_obs = stack_observations(observations)
        return self.current_obs

    def _step_envs(self, actions: np.ndarray):
        next_observations = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.float32)

        for env_idx, (env, action) in enumerate(zip(self.envs, actions)):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards[env_idx] = reward
            dones[env_idx] = float(done)
            if done:
                next_obs, _ = env.reset()
            next_observations.append(next_obs)

        return stack_observations(next_observations), rewards, dones

    def _step_envs_with_action_chunks(self, action_chunks: np.ndarray):
        next_observations = [None for _ in range(self.n_envs)]
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.float32)
        discounts = np.ones(self.n_envs, dtype=np.float32)
        active = np.ones(self.n_envs, dtype=bool)

        for exec_offset in range(self.action_exec_steps):
            action_idx = self.action_exec_index + exec_offset

            for env_idx, env in enumerate(self.envs):
                if not active[env_idx]:
                    continue

                next_obs, reward, terminated, truncated, _ = env.step(action_chunks[env_idx, action_idx])
                done = terminated or truncated

                rewards[env_idx] += (self.gamma ** exec_offset) * float(reward)
                self.num_timesteps += 1

                if done:
                    dones[env_idx] = 1.0
                    discounts[env_idx] = 0.0
                    next_obs, _ = env.reset()
                    active[env_idx] = False
                else:
                    discounts[env_idx] = self.gamma ** (exec_offset + 1)

                next_observations[env_idx] = next_obs

            if not active.any():
                break

        return stack_observations(next_observations), rewards, dones, discounts

    def collect_rollout(self) -> Dict[str, np.ndarray]:
        if self.current_obs is None:
            self.reset_envs()

        obs_buffer = {
            key: np.zeros((self.n_steps, self.n_envs, *value.shape[1:]), dtype=np.float32)
            for key, value in self.current_obs.items()
        }
        chains_buffer = np.zeros(
            (
                self.n_steps,
                self.n_envs,
                self.policy.denoising_steps + 1,
                self.policy.action_chunk_dim,
            ),
            dtype=np.float32,
        )
        log_probs_buffer = np.zeros(
            (self.n_steps, self.n_envs, self.policy.denoising_steps),
            dtype=np.float32,
        )
        values_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        rewards_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        dones_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        discounts_buffer = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        self.policy.eval()
        for step in range(self.n_steps):
            for key in obs_buffer:
                obs_buffer[key][step] = self.current_obs[key]

            with torch.no_grad():
                actions, chains, log_probs, values = self.policy.act(
                    self._obs_to_torch(self.current_obs),
                    deterministic=False,
                )

            action_chunks_np = actions.cpu().numpy()
            chains_buffer[step] = chains.cpu().numpy()
            log_probs_buffer[step] = log_probs.cpu().numpy()
            values_buffer[step] = values.cpu().numpy()

            next_obs, rewards, dones, discounts = self._step_envs_with_action_chunks(action_chunks_np)
            rewards_buffer[step] = rewards
            dones_buffer[step] = dones
            discounts_buffer[step] = discounts
            self.current_obs = next_obs

        with torch.no_grad():
            next_values = self.policy.value(self._obs_to_torch(self.current_obs)).cpu().numpy()

        if self.reward_normalizer is not None:
            rewards_buffer = self.reward_normalizer.normalize(rewards_buffer).astype(np.float32)

        advantages = np.zeros_like(rewards_buffer, dtype=np.float32)
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            next_discount = discounts_buffer[step]
            if step == self.n_steps - 1:
                next_values_step = next_values
            else:
                next_values_step = values_buffer[step + 1]

            delta = (
                rewards_buffer[step]
                + next_discount * next_values_step
                - values_buffer[step]
            )
            last_gae_lam = delta + next_discount * self.gae_lambda * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values_buffer

        return {
            "obs": obs_buffer,
            "chains": chains_buffer,
            "old_log_probs": log_probs_buffer,
            "old_values": values_buffer,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "discounts": discounts_buffer,
        }

    def update(self, rollout: Dict[str, np.ndarray]) -> DPPOMetrics:
        flat_obs = {
            key: value.reshape((self.n_steps * self.n_envs, *value.shape[2:]))
            for key, value in rollout["obs"].items()
        }
        flat_chains = rollout["chains"].reshape(
            self.n_steps * self.n_envs,
            self.policy.denoising_steps + 1,
            -1,
        )
        flat_old_log_probs = rollout["old_log_probs"].reshape(
            self.n_steps * self.n_envs,
            self.policy.denoising_steps,
        )
        flat_returns = rollout["returns"].reshape(-1)
        flat_advantages = rollout["advantages"].reshape(-1)

        if flat_advantages.size > 1:
            flat_advantages = (
                flat_advantages - flat_advantages.mean()
            ) / (flat_advantages.std() + 1e-8)

        n_env_steps = self.n_steps * self.n_envs
        n_total_steps = n_env_steps * self.policy.denoising_steps

        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        mae_losses = []

        self.policy.train()
        for _ in range(self.update_epochs):
            indices = np.random.permutation(n_total_steps)
            stop_update = False

            for start in range(0, n_total_steps, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                if batch_indices.size == 0:
                    continue

                obs_indices = batch_indices // self.policy.denoising_steps
                denoising_indices_np = batch_indices % self.policy.denoising_steps
                batch_obs = {
                    key: torch.as_tensor(value[obs_indices], dtype=torch.float32, device=self.device)
                    for key, value in flat_obs.items()
                }
                denoising_indices = torch.as_tensor(
                    denoising_indices_np,
                    dtype=torch.long,
                    device=self.device,
                )
                chains_prev = torch.as_tensor(
                    flat_chains[obs_indices, denoising_indices_np],
                    dtype=torch.float32,
                    device=self.device,
                )
                chains_next = torch.as_tensor(
                    flat_chains[obs_indices, denoising_indices_np + 1],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_log_probs = torch.as_tensor(
                    flat_old_log_probs[obs_indices, denoising_indices_np],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    flat_advantages[obs_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.as_tensor(
                    flat_returns[obs_indices],
                    dtype=torch.float32,
                    device=self.device,
                )

                values, log_probs, entropy = self.policy.evaluate_denoising_step(
                    batch_obs,
                    chains_prev,
                    chains_next,
                    denoising_indices,
                )

                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped_policy_loss = advantages * ratio
                clipped_policy_loss = advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                )
                policy_loss = -torch.min(unclipped_policy_loss, clipped_policy_loss).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                if self.mae_loss_coef > 0 and np.any(denoising_indices_np == 0):
                    mae_indices = denoising_indices_np == 0
                    mae_obs = {key: value[mae_indices] for key, value in batch_obs.items()}
                    mae_loss = self.policy.encoder.reconstruction_loss(mae_obs)
                else:
                    mae_loss = torch.zeros((), device=self.device)

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + self.mae_loss_coef * mae_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                    )

                losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
                mae_losses.append(mae_loss.item())

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        return DPPOMetrics(
            loss=float(np.mean(losses)),
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(approx_kls)),
            clip_fraction=float(np.mean(clip_fractions)),
            mae_loss=float(np.mean(mae_losses)),
        )

    def evaluate(
        self,
        eval_env,
        n_episodes: int = 5,
        deterministic: bool = True,
        gif_path: Optional[str] = None,
        gif_fps: int = 20,
        gif_max_frames: int = 300,
    ) -> float:
        episode_returns = []
        gif_frames = [] if gif_path and gif_max_frames > 0 else None
        self.policy.eval()

        for episode_idx in range(n_episodes):
            obs, _ = eval_env.reset(seed=1000 + episode_idx)
            done = False
            episode_return = 0.0
            capture_gif = gif_frames is not None and episode_idx == 0

            if capture_gif:
                frame = observation_to_eval_frame(obs)
                if frame is not None:
                    gif_frames.append(frame)

            while not done:
                obs_batch = {
                    key: value[None].astype(np.float32)
                    for key, value in obs.items()
                }
                with torch.no_grad():
                    action, _, _, _ = self.policy.act(
                        self._obs_to_torch(obs_batch),
                        deterministic=deterministic,
                    )
                action_chunk = action.cpu().numpy()[0]

                for exec_offset in range(self.action_exec_steps):
                    action_idx = self.action_exec_index + exec_offset
                    obs, reward, terminated, truncated, _ = eval_env.step(action_chunk[action_idx])
                    done = terminated or truncated
                    episode_return += float(reward)
                    if capture_gif and len(gif_frames) < gif_max_frames:
                        frame = observation_to_eval_frame(obs)
                        if frame is not None:
                            gif_frames.append(frame)
                    if done:
                        break

            episode_returns.append(episode_return)

        if gif_frames is not None and gif_path:
            if save_gif(gif_frames, gif_path, fps=gif_fps):
                print(f"[DPPO] eval GIF saved to {gif_path}")

        return float(np.mean(episode_returns))

    def save(self, path: str, config: Optional[Dict] = None) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
                "iteration": self.iteration,
                "config": config or {},
            },
            path,
        )

    def learn(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        checkpoint_dir: str = "./logs/checkpoints",
        final_model_path: str = "dppo_model_final.pt",
        n_eval_episodes: int = 5,
        eval_gif_dir: Optional[str] = None,
        eval_gif_fps: int = 20,
        eval_gif_max_frames: int = 300,
        swanlab_module=None,
        config: Optional[Dict] = None,
        progress_bar: bool = True,
    ) -> None:
        next_eval_step = (
            ((int(self.num_timesteps) // int(eval_freq)) + 1) * int(eval_freq)
            if eval_freq is not None and eval_freq > 0
            else None
        )
        next_save_step = (
            ((int(self.num_timesteps) // int(save_freq)) + 1) * int(save_freq)
            if save_freq is not None and save_freq > 0
            else None
        )
        progress = None
        if progress_bar:
            try:
                from tqdm.auto import tqdm

                progress = tqdm(
                    total=total_timesteps,
                    initial=min(self.num_timesteps, total_timesteps),
                    desc="DPPO",
                    unit="step",
                )
            except Exception as exc:
                print(f"[DPPO] progress bar disabled: {exc}")

        def print_status(message: str) -> None:
            if progress is not None:
                progress.write(message)
            else:
                print(message)

        while self.num_timesteps < total_timesteps:
            previous_timesteps = self.num_timesteps
            rollout = self.collect_rollout()
            metrics = self.update(rollout)
            self.iteration += 1

            mean_reward = float(np.mean(np.sum(rollout["rewards"], axis=0)))
            log_data = {
                "train/loss": metrics.loss,
                "train/policy_loss": metrics.policy_loss,
                "train/value_loss": metrics.value_loss,
                "train/entropy": metrics.entropy,
                "train/approx_kl": metrics.approx_kl,
                "train/clip_fraction": metrics.clip_fraction,
                "train/mae_loss": metrics.mae_loss,
                "rollout/mean_reward": mean_reward,
                "time/num_timesteps": self.num_timesteps,
                "time/iteration": self.iteration,
            }

            if progress is not None:
                progress.update(
                    max(
                        min(self.num_timesteps, total_timesteps)
                        - min(previous_timesteps, total_timesteps),
                        0,
                    )
                )
                progress.set_postfix(
                    loss=f"{metrics.loss:.4f}",
                    reward=f"{mean_reward:.4f}",
                    kl=f"{metrics.approx_kl:.5f}",
                )

            print_status(
                f"[DPPO] itr={self.iteration} steps={self.num_timesteps} "
                f"loss={metrics.loss:.4f} reward={mean_reward:.4f} "
                f"kl={metrics.approx_kl:.5f}"
            )

            if (
                eval_env is not None
                and next_eval_step is not None
                and self.num_timesteps >= next_eval_step
            ):
                gif_path = None
                if eval_gif_dir:
                    gif_path = os.path.join(eval_gif_dir, f"eval_step_{self.num_timesteps}.gif")

                eval_return = self.evaluate(
                    eval_env,
                    n_episodes=n_eval_episodes,
                    deterministic=True,
                    gif_path=gif_path,
                    gif_fps=eval_gif_fps,
                    gif_max_frames=eval_gif_max_frames,
                )
                log_data["eval/mean_return"] = eval_return
                print_status(f"[DPPO] eval_return={eval_return:.4f}")
                next_eval_step += eval_freq

            if next_save_step is not None and self.num_timesteps >= next_save_step:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"dppo_model_step_{self.num_timesteps}.pt",
                )
                checkpoint_path = avoid_existing_path(checkpoint_path)
                self.save(checkpoint_path, config=config)
                next_save_step += save_freq

            if swanlab_module is not None:
                swanlab_module.log(log_data, step=self.num_timesteps)

        if progress is not None:
            progress.close()

        self.save(final_model_path, config=config)
