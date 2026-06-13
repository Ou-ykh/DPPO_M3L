from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build vetted OmniReset banks for RLEnv")
    parser.add_argument("--xml_path", type=str, default="./mjmodel.xml")
    parser.add_argument(
        "--reset_types",
        type=str,
        default="reaching,near_object,stable_grasp,near_goal",
    )
    parser.add_argument("--bank_size_per_type", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default="./logs/reset_banks")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--force_rebuild", type=str2bool, default=True)
    parser.add_argument("--reset_stabilize_steps", type=int, default=64)
    parser.add_argument("--reset_max_sample_attempts", type=int, default=1536)
    parser.add_argument("--reset_grasp_close_steps", type=int, default=20)
    parser.add_argument("--reset_goal_offset_bank_size", type=int, default=64)
    parser.add_argument("--reset_goal_offset_perturb_steps", type=int, default=12)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--mujoco_gl", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_idx))
    mujoco_gl = args.mujoco_gl or ("glfw" if os.name == "nt" else "egl")
    os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    if mujoco_gl == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.gpu_idx))

    from RLenv import RLEnv

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = RLEnv(
        xml_path=args.xml_path,
        reset_types=args.reset_types,
        reset_bank_size_per_type=args.bank_size_per_type,
        reset_cache_dir=args.cache_dir,
        reset_force_rebuild=args.force_rebuild,
        reset_bank_verbose=True,
        reset_randomize_on_reset=False,
        reset_stabilize_steps=args.reset_stabilize_steps,
        reset_max_sample_attempts=args.reset_max_sample_attempts,
        reset_grasp_close_steps=args.reset_grasp_close_steps,
        reset_goal_offset_bank_size=args.reset_goal_offset_bank_size,
        reset_goal_offset_perturb_steps=args.reset_goal_offset_perturb_steps,
    )

    try:
        bank = env._ensure_reset_bank()
        sizes = {key: int(value["qpos"].shape[0]) for key, value in bank.items()}
        print(f"[BUILD-RESET-BANK] complete sizes={sizes}", flush=True)
        print(f"[BUILD-RESET-BANK] cache_root={Path(args.cache_dir) / Path(args.xml_path).stem}", flush=True)
        for reset_type, entry in bank.items():
            print(
                f"[BUILD-RESET-BANK] {reset_type}: keys={sorted(entry.keys())} "
                f"arm_qpos_shape={entry.get('arm_qpos', np.empty((0,))).shape} "
                f"gripper_qpos_shape={entry.get('gripper_qpos', np.empty((0,))).shape}",
                flush=True,
            )
    finally:
        renderer = getattr(env, "renderer", None)
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
