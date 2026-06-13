from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from models.dppo_mae import MAEVitEncoder


class DirectBCPolicy(nn.Module):
    """Direct action-regression policy using the existing MAE observation encoder."""

    def __init__(
        self,
        mae_model: nn.Module,
        dim_embeddings: int,
        action_dim: int,
        action_horizon: int,
        frame_stack: int,
        vision_only_control: bool,
        actor_hidden_dim: int,
        actor_depth: int,
        actor_dropout: float,
        action_output: str,
    ) -> None:
        super().__init__()
        if actor_depth < 1:
            raise ValueError("--actor_depth must be at least 1")
        if action_output not in {"tanh", "clamp", "none"}:
            raise ValueError("--action_output must be one of: tanh, clamp, none")

        self.encoder = MAEVitEncoder(
            mae_model=mae_model,
            dim_embeddings=dim_embeddings,
            frame_stack=frame_stack,
            vision_only_control=vision_only_control,
        )
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.action_chunk_dim = self.action_dim * self.action_horizon
        self.action_output = action_output

        layers: List[nn.Module] = []
        in_dim = dim_embeddings
        for _ in range(actor_depth):
            layers.append(nn.Linear(in_dim, actor_hidden_dim))
            layers.append(nn.SiLU())
            if actor_dropout > 0.0:
                layers.append(nn.Dropout(actor_dropout))
            in_dim = actor_hidden_dim
        self.actor_body = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, self.action_chunk_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encoder(observations)
        hidden = self.actor_body(features)
        actions = self.action_head(hidden)
        actions = actions.reshape(actions.shape[0], self.action_horizon, self.action_dim)
        if self.action_output == "tanh":
            actions = torch.tanh(actions)
        elif self.action_output == "clamp":
            actions = torch.clamp(actions, -1.0, 1.0)
        return actions

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, None, None, torch.Tensor]:
        actions = self.forward(observations)
        values = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        return actions, None, None, values


def _extract_state_dict(
    checkpoint_obj,
    preferred_keys: Sequence[str],
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
    return state_dict


def load_direct_checkpoint(
    policy: DirectBCPolicy,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(
        checkpoint,
        preferred_keys=("policy_state_dict", "state_dict"),
    )
    incompatible = policy.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        f"[DirectBC] loaded policy checkpoint from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[DirectBC] first missing keys: {missing[:10]}")
    if unexpected:
        print(f"[DirectBC] first unexpected keys: {unexpected[:10]}")
    return checkpoint if isinstance(checkpoint, dict) else {}
