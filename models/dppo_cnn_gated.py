from typing import Dict, Optional, Tuple

import torch
from torch import nn

from models.dppo_mae import GaussianDiffusionActor


def _to_channels_first(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 5:
        value = value.reshape(value.shape[0], value.shape[1], value.shape[2], value.shape[3], -1)
    if value.dim() != 4:
        raise ValueError(f"expected 4D observation tensor, got shape {tuple(value.shape)}")
    if value.shape[1] <= 64 and value.shape[1] < value.shape[-1]:
        return value
    return value.permute(0, 3, 1, 2).contiguous()


class ConvEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim_embeddings: int,
        base_channels: int = 32,
    ):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        layers = []
        prev_channels = in_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(1, out_channels),
                    nn.SiLU(),
                ]
            )
            prev_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], dim_embeddings),
            nn.LayerNorm(dim_embeddings),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(value))


class GatedCNNEncoder(nn.Module):
    def __init__(
        self,
        dim_embeddings: int,
        frame_stack: int,
        vision_only_control: bool = False,
        image_channels_per_frame: int = 6,
        tactile_channels_per_frame: int = 6,
        base_channels: int = 32,
        gate_init_bias: float = -4.0,
        gate_min_value: float = 0.0,
    ):
        super().__init__()
        self.frame_stack = int(frame_stack)
        self.vision_only_control = bool(vision_only_control)
        self.register_buffer(
            "gate_min_value",
            torch.tensor(float(max(0.0, min(1.0, gate_min_value))), dtype=torch.float32),
            persistent=True,
        )
        image_channels = image_channels_per_frame * self.frame_stack
        tactile_channels = tactile_channels_per_frame * self.frame_stack
        self.image_encoder = ConvEncoder2D(
            in_channels=image_channels,
            dim_embeddings=dim_embeddings,
            base_channels=base_channels,
        )
        self.tactile_encoder = ConvEncoder2D(
            in_channels=tactile_channels,
            dim_embeddings=dim_embeddings,
            base_channels=base_channels,
        )
        self.tactile_proj = nn.Sequential(
            nn.Linear(dim_embeddings, dim_embeddings),
            nn.SiLU(),
            nn.Linear(dim_embeddings, dim_embeddings),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(dim_embeddings * 2, dim_embeddings),
            nn.SiLU(),
            nn.Linear(dim_embeddings, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, gate_init_bias)
        self.last_gate: Optional[torch.Tensor] = None

    def set_gate_min(self, value: float) -> None:
        self.gate_min_value.fill_(float(max(0.0, min(1.0, value))))

    def get_gate_min(self) -> float:
        return float(self.gate_min_value.detach().item())

    def _prepare_image(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["image"]
        image = image.to(dtype=torch.float32)
        if float(image.max().detach().item()) > 1.5:
            image = image / 255.0
        return _to_channels_first(image).clamp(0.0, 1.0)

    def _prepare_tactile(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.vision_only_control or "tactile" not in observations:
            return None
        tactile = observations["tactile"].to(dtype=torch.float32)
        if tactile.dim() == 5:
            tactile = tactile.reshape(tactile.shape[0], -1, tactile.shape[-2], tactile.shape[-1])
        if float(tactile.min().detach().item()) < -0.05:
            tactile = (tactile + 1.0) / 2.0
        elif float(tactile.max().detach().item()) > 1.5:
            tactile = tactile / 255.0
        return tactile.clamp(0.0, 1.0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_img = self.image_encoder(self._prepare_image(observations))
        tactile = self._prepare_tactile(observations)
        if tactile is None:
            self.last_gate = None
            return z_img

        z_tac = self.tactile_encoder(tactile)
        tactile_delta = self.tactile_proj(z_tac)
        gate = torch.sigmoid(self.gate_net(torch.cat((z_img, z_tac), dim=-1)))
        gate_min = self.gate_min_value.to(device=gate.device, dtype=gate.dtype).clamp(0.0, 1.0)
        gate = gate_min + (1.0 - gate_min) * gate
        self.last_gate = gate.detach()
        return z_img + gate * tactile_delta

    def reconstruction_loss(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["image"]
        return torch.zeros((), dtype=torch.float32, device=image.device)


class GatedCNNDPPOPolicy(nn.Module):
    def __init__(
        self,
        dim_embeddings: int,
        action_dim: int,
        action_horizon: int,
        frame_stack: int,
        vision_only_control: bool = False,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        encoder_base_channels: int = 32,
        gate_init_bias: float = -4.0,
        gate_min_value: float = 0.0,
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
        self.encoder = GatedCNNEncoder(
            dim_embeddings=dim_embeddings,
            frame_stack=frame_stack,
            vision_only_control=vision_only_control,
            base_channels=encoder_base_channels,
            gate_init_bias=gate_init_bias,
            gate_min_value=gate_min_value,
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

    def set_gate_min(self, value: float) -> None:
        self.encoder.set_gate_min(value)

    def get_gate_min(self) -> float:
        return self.encoder.get_gate_min()

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
