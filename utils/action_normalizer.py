from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ActionNormalizer:
    mode: str = "none"
    scale: Optional[np.ndarray] = None
    min_scale: float = 1e-6

    @classmethod
    def identity(cls, action_dim: Optional[int] = None) -> "ActionNormalizer":
        scale = None if action_dim is None else np.ones(int(action_dim), dtype=np.float32)
        return cls(mode="none", scale=scale)

    @classmethod
    def from_dict(
        cls,
        payload: Optional[Dict[str, Any]],
        action_dim: Optional[int] = None,
    ) -> "ActionNormalizer":
        if isinstance(payload, ActionNormalizer):
            return payload
        if not isinstance(payload, dict):
            return cls.identity(action_dim)

        mode = str(payload.get("mode", "none")).lower()
        if mode in {"none", "identity"}:
            return cls.identity(action_dim)
        if mode != "max_abs":
            raise ValueError(f"unsupported action normalizer mode '{mode}'")

        scale = np.asarray(payload.get("scale"), dtype=np.float32).reshape(-1)
        if action_dim is not None and scale.size != int(action_dim):
            raise ValueError(
                f"action normalizer scale dim {scale.size} != action_dim {int(action_dim)}"
            )
        if scale.size == 0:
            raise ValueError("action normalizer scale is empty")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("action normalizer scale must be positive and finite")
        min_scale = float(payload.get("min_scale", 1e-6))
        return cls(mode=mode, scale=scale.astype(np.float32), min_scale=min_scale)

    @classmethod
    def from_max_abs(
        cls,
        max_abs: np.ndarray,
        *,
        min_scale: float = 1e-6,
    ) -> "ActionNormalizer":
        scale = np.asarray(max_abs, dtype=np.float32).reshape(-1)
        if scale.size == 0:
            raise ValueError("max_abs is empty")
        if not np.all(np.isfinite(scale)):
            raise ValueError("max_abs must be finite")
        min_scale = max(float(min_scale), 0.0)
        scale = np.maximum(scale, 0.0)
        inactive = scale < min_scale
        scale[inactive] = 1.0
        return cls(mode="max_abs", scale=scale.astype(np.float32), min_scale=min_scale)

    @property
    def enabled(self) -> bool:
        return self.mode == "max_abs" and self.scale is not None

    def to_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"mode": "none"}
        return {
            "mode": self.mode,
            "scale": [float(value) for value in np.asarray(self.scale).reshape(-1)],
            "min_scale": float(self.min_scale),
        }

    def _scale_np(self, actions: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.ones((actions.shape[-1],), dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if actions.shape[-1] != scale.size:
            raise ValueError(f"action dim {actions.shape[-1]} != normalizer dim {scale.size}")
        return scale

    def _scale_torch(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.ones((actions.shape[-1],), dtype=actions.dtype, device=actions.device)
        scale_np = np.asarray(self.scale, dtype=np.float32)
        if actions.shape[-1] != scale_np.size:
            raise ValueError(f"action dim {actions.shape[-1]} != normalizer dim {scale_np.size}")
        return torch.as_tensor(scale_np, dtype=actions.dtype, device=actions.device)

    def normalize_np(self, actions: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if self.enabled:
            values = values / self._scale_np(values)
        if clip is not None and math.isfinite(float(clip)) and float(clip) > 0.0:
            values = np.clip(values, -float(clip), float(clip))
        return values.astype(np.float32, copy=False)

    def denormalize_np(self, actions: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if clip is not None and math.isfinite(float(clip)) and float(clip) > 0.0:
            values = np.clip(values, -float(clip), float(clip))
        if self.enabled:
            values = values * self._scale_np(values)
        return values.astype(np.float32, copy=False)

    def normalize_torch(self, actions: torch.Tensor, clip: Optional[float] = None) -> torch.Tensor:
        values = actions
        if self.enabled:
            values = values / self._scale_torch(values)
        if clip is not None and math.isfinite(float(clip)) and float(clip) > 0.0:
            values = values.clamp(-float(clip), float(clip))
        return values

    def denormalize_torch(self, actions: torch.Tensor, clip: Optional[float] = None) -> torch.Tensor:
        values = actions
        if clip is not None and math.isfinite(float(clip)) and float(clip) > 0.0:
            values = values.clamp(-float(clip), float(clip))
        if self.enabled:
            values = values * self._scale_torch(values)
        return values

    def summary(self) -> Dict[str, float]:
        if not self.enabled:
            return {"enabled": 0.0}
        scale = np.asarray(self.scale, dtype=np.float32)
        return {
            "enabled": 1.0,
            "scale_min": float(np.min(scale)),
            "scale_max": float(np.max(scale)),
            "scale_mean": float(np.mean(scale)),
            "active_dims": float(np.sum(scale != 1.0)),
        }
