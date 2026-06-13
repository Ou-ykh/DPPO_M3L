from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser("Single-GPU MJWarp rollout worker for DPPO")
    parser.add_argument("--xml_path", required=True)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--policy_config", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_envs", type=int, required=True)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--action_exec_start", type=int, default=0)
    parser.add_argument("--action_exec_steps", type=int, default=1)
    parser.add_argument("--sim_substeps", type=int, default=25)
    parser.add_argument("--reward_success_bonus", type=float, default=1000.0)
    parser.add_argument("--reward_distance_coef", type=float, default=5.0)
    parser.add_argument("--reward_time_penalty", type=float, default=0.5)
    parser.add_argument("--reward_gripper_coef", type=float, default=0.2)
    parser.add_argument("--reward_smoothness_coef", type=float, default=0.0)
    parser.add_argument("--reward_truncation_penalty", type=float, default=0.0)
    parser.add_argument("--reset_types", default="reaching,near_object,stable_grasp,near_goal")
    parser.add_argument("--reset_stabilize_steps", type=int, default=32)
    parser.add_argument("--reset_max_sample_attempts", type=int, default=64)
    parser.add_argument("--reset_bank_size_per_type", type=int, default=64)
    parser.add_argument("--reset_cache_dir", default=None)
    parser.add_argument("--reset_force_rebuild", action="store_true")
    parser.add_argument("--reset_grasp_close_steps", type=int, default=20)
    parser.add_argument("--reset_goal_offset_bank_size", type=int, default=64)
    parser.add_argument("--reset_goal_offset_perturb_steps", type=int, default=12)
    parser.add_argument("--arm_action_scale", type=float, default=0.1)
    parser.add_argument("--gripper_action_scale", type=float, default=1.0)
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--no_normalize_assets", action="store_true")
    parser.add_argument("--worker_index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch

    from utils.mjwarp_dppo import (
        MJWarpStateRollout,
        read_policy_config,
        save_rollout_npz,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"worker: {args.worker_index}")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    print("torch device:", device)
    if torch.cuda.is_available():
        print("torch gpu:", torch.cuda.get_device_name(0))

    policy_config = read_policy_config(args.policy_config)
    policy_kind = policy_config.pop("policy_kind", "dppo")
    if policy_kind == "mlp":
        from models.ppo_mlp import StateMLPActorCritic

        policy = StateMLPActorCritic(**policy_config).to(device)
    else:
        from models.dppo_mae import DPPOStatePolicy

        policy = DPPOStatePolicy(**policy_config).to(device)
    checkpoint = torch.load(args.policy_path, map_location="cpu")
    state_dict = checkpoint.get("policy_state_dict", checkpoint.get("state_dict", checkpoint))
    policy.load_state_dict(state_dict, strict=True)

    rollout_env = MJWarpStateRollout(
        xml_path=args.xml_path,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        seed=args.seed,
        nconmax=args.nconmax,
        naconmax=args.naconmax,
        njmax=args.njmax,
        gamma=args.gamma,
        action_exec_index=args.action_exec_start,
        action_exec_steps=args.action_exec_steps,
        sim_substeps=args.sim_substeps,
        reward_success_bonus=args.reward_success_bonus,
        reward_distance_coef=args.reward_distance_coef,
        reward_time_penalty=args.reward_time_penalty,
        reward_gripper_coef=args.reward_gripper_coef,
        reward_smoothness_coef=args.reward_smoothness_coef,
        reward_truncation_penalty=args.reward_truncation_penalty,
        reset_types=args.reset_types,
        reset_stabilize_steps=args.reset_stabilize_steps,
        reset_max_sample_attempts=args.reset_max_sample_attempts,
        reset_bank_size_per_type=args.reset_bank_size_per_type,
        reset_cache_dir=args.reset_cache_dir,
        reset_force_rebuild=args.reset_force_rebuild,
        reset_grasp_close_steps=args.reset_grasp_close_steps,
        reset_goal_offset_bank_size=args.reset_goal_offset_bank_size,
        reset_goal_offset_perturb_steps=args.reset_goal_offset_perturb_steps,
        arm_action_scale=args.arm_action_scale,
        gripper_action_scale=args.gripper_action_scale,
        repo_root=args.repo_root,
        normalize_assets=not args.no_normalize_assets,
    )

    print("state_dim:", rollout_env.state_dim)
    print("action_dim:", rollout_env.action_dim)
    print("rewritten asset paths:", rollout_env.rewritten_assets)
    print("reset bank sizes:", rollout_env.reset_bank_sizes)
    print("reset bank cache:", rollout_env.reset_bank_cache_path)
    print("arm action scale:", rollout_env.arm_action_scale)
    print("gripper action scale:", rollout_env.gripper_action_scale)
    print("reward success bonus:", rollout_env.reward_success_bonus)
    print("reward smoothness coef:", rollout_env.reward_smoothness_coef)
    print("reward truncation penalty:", rollout_env.reward_truncation_penalty)
    start = time.perf_counter()
    rollout = rollout_env.collect(policy, device=device)
    elapsed = time.perf_counter() - start
    transitions = args.n_envs * args.n_steps * args.action_exec_steps
    print(f"rollout seconds: {elapsed:.4f}")
    print(f"env transitions/sec: {transitions / max(elapsed, 1e-9):.2f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_rollout_npz(output_path, rollout)
    print("saved rollout:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
