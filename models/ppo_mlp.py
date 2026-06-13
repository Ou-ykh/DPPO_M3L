from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: Optional[int] = None,
    activation=nn.SiLU,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError(f"MLP depth must be >= 1, got {depth}")

    layers = []
    in_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        in_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class StateMLPActorCritic(nn.Module):
    """Simple Gaussian MLP actor-critic for state-only PPO teachers."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 1,
        hidden_dim: int = 256,
        depth: int = 3,
        log_std_init: float = -2.5,
    ):
        super().__init__()
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.action_chunk_dim = self.action_dim * self.action_horizon

        self.actor = _make_mlp(
            input_dim=self.state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            output_dim=self.action_chunk_dim,
        )
        self.critic = _make_mlp(
            input_dim=self.state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            output_dim=1,
        )
        self.log_std = nn.Parameter(torch.full((self.action_chunk_dim,), float(log_std_init)))
        self.obs_clip = 1e3
        self._init_actor_head()

    def _init_actor_head(self) -> None:
        final_linear = None
        for module in self.actor.modules():
            if isinstance(module, nn.Linear):
                final_linear = module
        if final_linear is not None:
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _state(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = observations["state"]
        state = torch.nan_to_num(
            state,
            nan=0.0,
            posinf=self.obs_clip,
            neginf=-self.obs_clip,
        )
        return torch.clamp(state, -self.obs_clip, self.obs_clip)

    def distribution(self, observations: Dict[str, torch.Tensor]) -> torch.distributions.Normal:
        state = self._state(observations)
        mean = torch.tanh(self.actor(state))
        log_std = torch.clamp(self.log_std, min=-5.0, max=0.0)
        std = torch.exp(log_std).expand_as(mean).clamp_min(1e-6)
        if not torch.isfinite(mean).all():
            raise RuntimeError("policy actor produced non-finite mean")
        return torch.distributions.Normal(mean, std)

    def value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._state(observations)).squeeze(-1)

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        if deterministic:
            flat_actions = distribution.mean
        else:
            flat_actions = distribution.rsample()
        flat_actions = torch.clamp(flat_actions, -1.0, 1.0)
        log_probs = distribution.log_prob(flat_actions).sum(dim=-1)
        values = self.value(observations)
        actions = flat_actions.reshape(-1, self.action_horizon, self.action_dim)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_actions = actions.reshape(actions.shape[0], self.action_chunk_dim)
        distribution = self.distribution(observations)
        log_probs = distribution.log_prob(flat_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.value(observations)
        return values, log_probs, entropy


@dataclass
class PPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    mae_loss: float = 0.0


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


class PPOTrainer:
    def __init__(
        self,
        policy: StateMLPActorCritic,
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
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        norm_reward: bool = False,
        value_clip_range: Optional[float] = None,
        return_clip: Optional[float] = None,
        action_exec_index: int = 0,
        action_exec_steps: int = 1,
    ):
        self.policy = policy.to(device)
        self.envs = envs
        self.device = device
        self.n_envs = len(envs)
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.update_epochs = int(update_epochs)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.reward_normalizer = RewardNormalizer() if norm_reward else None
        self.value_clip_range = None if value_clip_range is None else float(value_clip_range)
        self.return_clip = None if return_clip is None else float(return_clip)

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

        self.action_exec_index = int(action_exec_index)
        self.action_exec_steps = int(action_exec_steps)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate)
        self.num_timesteps = 0
        self.iteration = 0

    def update(self, rollout: Dict[str, np.ndarray]) -> PPOMetrics:
        flat_obs = {
            key: value.reshape((self.n_steps * self.n_envs, *value.shape[2:]))
            for key, value in rollout["obs"].items()
        }
        flat_actions = rollout["actions"].reshape(
            self.n_steps * self.n_envs,
            self.policy.action_horizon,
            self.policy.action_dim,
        )
        flat_old_log_probs = rollout["old_log_probs"].reshape(-1)
        flat_old_values = rollout["old_values"].reshape(-1)
        flat_returns = rollout["returns"].reshape(-1)
        flat_advantages = rollout["advantages"].reshape(-1)

        if self.return_clip is not None and self.return_clip > 0:
            flat_returns = np.clip(flat_returns, -self.return_clip, self.return_clip)

        if flat_advantages.size > 1:
            flat_advantages = (
                flat_advantages - flat_advantages.mean()
            ) / (flat_advantages.std() + 1e-8)

        valid_mask = np.ones(self.n_steps * self.n_envs, dtype=bool)
        for value in flat_obs.values():
            valid_mask &= np.isfinite(value.reshape(value.shape[0], -1)).all(axis=1)
        valid_mask &= np.isfinite(flat_actions.reshape(flat_actions.shape[0], -1)).all(axis=1)
        valid_mask &= np.isfinite(flat_old_log_probs)
        valid_mask &= np.isfinite(flat_old_values)
        valid_mask &= np.isfinite(flat_returns)
        valid_mask &= np.isfinite(flat_advantages)

        num_invalid = int((~valid_mask).sum())
        if num_invalid:
            print(
                f"[PPO-MLP] filtered {num_invalid} / {valid_mask.size} non-finite rollout samples",
                flush=True,
            )
            flat_obs = {key: value[valid_mask] for key, value in flat_obs.items()}
            flat_actions = flat_actions[valid_mask]
            flat_old_log_probs = flat_old_log_probs[valid_mask]
            flat_old_values = flat_old_values[valid_mask]
            flat_returns = flat_returns[valid_mask]
            flat_advantages = flat_advantages[valid_mask]

        n_samples = int(flat_actions.shape[0])
        if n_samples == 0:
            raise RuntimeError("PPO update received no finite rollout samples")

        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []

        self.policy.train()
        for _ in range(self.update_epochs):
            indices = np.random.permutation(n_samples)
            stop_update = False

            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                if batch_indices.size == 0:
                    continue

                batch_obs = {
                    key: torch.as_tensor(value[batch_indices], dtype=torch.float32, device=self.device)
                    for key, value in flat_obs.items()
                }
                actions = torch.as_tensor(
                    flat_actions[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_log_probs = torch.as_tensor(
                    flat_old_log_probs[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_values = torch.as_tensor(
                    flat_old_values[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    flat_advantages[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.as_tensor(
                    flat_returns[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )

                values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, actions)
                if not (
                    torch.isfinite(values).all()
                    and torch.isfinite(log_probs).all()
                    and torch.isfinite(entropy).all()
                ):
                    continue
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped_policy_loss = advantages * ratio
                clipped_policy_loss = advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                )
                policy_loss = -torch.min(unclipped_policy_loss, clipped_policy_loss).mean()
                if self.value_clip_range is not None and self.value_clip_range > 0:
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.value_clip_range,
                        self.value_clip_range,
                    )
                    value_loss_unclipped = (values - returns).pow(2)
                    value_loss_clipped = (values_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_loss_unclipped,
                        value_loss_clipped,
                    ).mean()
                else:
                    value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if any(not torch.isfinite(param).all() for param in self.policy.parameters()):
                    raise RuntimeError("policy parameters became non-finite during PPO update")

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

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        if not losses:
            raise RuntimeError("PPO update skipped every minibatch due to non-finite values")

        return PPOMetrics(
            loss=float(np.mean(losses)),
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(approx_kls)),
            clip_fraction=float(np.mean(clip_fractions)),
            mae_loss=0.0,
        )

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
                "policy_kind": "mlp",
                "config": config or {},
            },
            path,
        )
