"""Probe whether an MJCF model can be loaded and stepped by MJX.

This script is intentionally small and side-effect free: it does not modify the
XML file. Run it inside the conda environment that has jax and mujoco-mjx
installed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _rewrite_asset_paths(xml_path: Path, repo_root: Path) -> tuple[str, int]:
    """Rewrites Windows/relative asset file paths to absolute local paths."""
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
            # Unknown Windows absolute path. Leave it as-is so MuJoCo reports a
            # clear missing-file error instead of guessing the wrong location.
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


def _block_until_ready(tree):
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _print_model_summary(model) -> None:
    attrs = [
        "nq",
        "nv",
        "nu",
        "nbody",
        "njnt",
        "ngeom",
        "nmesh",
        "ntendon",
        "neq",
        "nsensor",
        "ncam",
        "nsite",
        "nconmax",
        "njmax",
    ]
    for attr in attrs:
        print(f"{attr}: {getattr(model, attr)}")


def _broadcast_data(data, batch_size: int):
    import jax
    import jax.numpy as jnp

    if batch_size == 1:
        return data, None

    # MJX-Warp keeps contact/constraint buffer metadata in data._impl. These
    # leaves are not ordinary per-world state arrays; adding a leading batch
    # dimension to them triggers FFI shape assertions such as contact__dim
    # expected ndim 1 but got ndim 2. Batch only public Data fields and leave
    # _impl as shared metadata for the vmapped Warp call.
    impl = getattr(data, "_impl", None)

    def maybe_broadcast(value):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return jnp.broadcast_to(value, (batch_size,) + value.shape)
        return value

    batched = jax.tree_util.tree_map(maybe_broadcast, data)
    data_axes = jax.tree_util.tree_map(lambda _: 0, batched)

    if impl is not None:
        batched = batched.replace(_impl=impl)
        data_axes = data_axes.replace(_impl=None)

    return batched, data_axes


def _probe_impl(
    mj_model,
    impl: str,
    steps: int,
    batch_size: int,
    nconmax: int | None,
    naconmax: int,
    njmax: int,
) -> None:
    if impl == "warp":
        return _probe_mjwarp_impl(mj_model, steps, batch_size, nconmax, naconmax, njmax)

    return _probe_mjx_impl(mj_model, impl, steps, batch_size, naconmax, njmax)


def _probe_mjwarp_impl(
    mj_model,
    steps: int,
    batch_size: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
) -> None:
    try:
        import mujoco_warp as mjw
        import warp as wp
    except ModuleNotFoundError as exc:
        if exc.name == "mujoco_warp":
            raise RuntimeError(
                "Missing native MJWarp package. Install it in this environment with "
                "`pip install mujoco-warp`, then rerun with `--impl warp`. "
                "The pip package uses a hyphen, but the Python import is `mujoco_warp`."
            ) from exc
        raise

    print("\n=== MJWarp native ===")
    print("note: using mjw.make_data(..., nworld=batch_size), not jax.vmap")

    make_kwargs = {"nworld": batch_size, "njmax": njmax}
    if naconmax is not None:
        make_kwargs["naconmax"] = naconmax
    elif nconmax is not None:
        make_kwargs["nconmax"] = nconmax

    model = mjw.put_model(mj_model)
    data = mjw.make_data(mj_model, **make_kwargs)

    def run() -> None:
        for _ in range(steps):
            mjw.step(model, data)
        wp.synchronize()

    # First call includes Warp kernel compilation/capture overhead.
    t0 = time.perf_counter()
    run()
    compile_and_step_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    run()
    runtime = time.perf_counter() - t1

    print(f"warp device: {wp.get_device()}")
    print(f"batch size: {batch_size}")
    print(f"steps: {steps}")
    print(f"nconmax/world: {nconmax}")
    print(f"naconmax/total: {naconmax}")
    print(f"njmax/world: {njmax}")
    print(f"compile+run seconds: {compile_and_step_time:.4f}")
    print(f"run seconds: {runtime:.4f}")
    print(f"env steps/sec: {(batch_size * steps / runtime):.2f}")
    print("qpos shape:", data.qpos.shape)
    print("qvel shape:", data.qvel.shape)
    print("ok")


def _probe_mjx_impl(
    mj_model,
    impl: str,
    steps: int,
    batch_size: int,
    naconmax: int | None,
    njmax: int,
) -> None:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    print(f"\n=== MJX impl={impl} ===")
    put_kwargs = {}
    make_kwargs = {}

    mx = mjx.put_model(mj_model, **put_kwargs)
    data = mjx.make_data(mj_model, **make_kwargs)

    if mj_model.nu:
        data = data.replace(ctrl=jnp.zeros((mj_model.nu,), dtype=data.ctrl.dtype))
    data, data_axes = _broadcast_data(data, batch_size)

    def rollout_single(d):
        for _ in range(steps):
            d = mjx.step(mx, d)
        return d

    if batch_size == 1:
        run = jax.jit(rollout_single)
    else:
        run = jax.jit(jax.vmap(rollout_single, in_axes=(data_axes,)))

    # First call includes compilation.
    t0 = time.perf_counter()
    data = run(data)
    _block_until_ready(data)
    compile_and_step_time = time.perf_counter() - t0

    # Second call is closer to runtime speed.
    t1 = time.perf_counter()
    data = run(data)
    _block_until_ready(data)
    runtime = time.perf_counter() - t1

    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print(f"batch size: {batch_size}")
    print(f"steps: {steps}")
    print(f"compile+run seconds: {compile_and_step_time:.4f}")
    print(f"run seconds: {runtime:.4f}")
    print(f"env steps/sec: {(batch_size * steps / runtime):.2f}")
    print("qpos shape:", data.qpos.shape)
    print("qvel shape:", data.qvel.shape)
    print("ok")


def _parse_devices(raw: str) -> list[str]:
    devices = [device.strip() for device in raw.split(",") if device.strip()]
    if not devices:
        raise ValueError("device list is empty")
    return devices


def _build_worker_command(args, script_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        args.xml,
        "--impl",
        args.impl,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--njmax",
        str(args.njmax),
    ]

    if args.nconmax is not None:
        command.extend(["--nconmax", str(args.nconmax)])
    if args.naconmax is not None:
        command.extend(["--naconmax", str(args.naconmax)])
    if args.repo_root is not None:
        command.extend(["--repo-root", args.repo_root])
    if args.no_normalize_assets:
        command.append("--no-normalize-assets")

    return command


def _run_multi_gpu(args) -> int:
    sim_devices = _parse_devices(args.sim_devices)
    script_path = Path(__file__).resolve()
    base_command = _build_worker_command(args, script_path)

    print("=== Multi-GPU MJWarp coordinator ===")
    print("sim devices:", ",".join(sim_devices))
    print("update device:", args.update_device if args.update_device is not None else "(none)")
    print("per-device batch size:", args.batch_size)
    print("total rollout envs:", args.batch_size * len(sim_devices))
    print("steps:", args.steps)
    print("worker command:", " ".join(base_command))

    processes = []
    for index, device in enumerate(sim_devices):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        command = base_command + ["--worker-index", str(index)]
        print(f"\nlaunch worker {index}: physical GPU {device}")
        processes.append(
            (
                index,
                device,
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                ),
            )
        )

    ok = True
    total_env_steps_per_sec = 0.0
    for index, device, process in processes:
        output, _ = process.communicate()
        print(f"\n===== worker {index} physical GPU {device} output =====")
        print(output, end="" if output.endswith("\n") else "\n")

        if process.returncode != 0:
            ok = False

        match = re.search(r"env steps/sec:\s*([0-9.]+)", output)
        if match:
            total_env_steps_per_sec += float(match.group(1))

    print("\n=== Multi-GPU summary ===")
    print("workers:", len(sim_devices))
    print("per-device batch size:", args.batch_size)
    print("total rollout envs:", args.batch_size * len(sim_devices))
    print(f"summed env steps/sec: {total_env_steps_per_sec:.2f}")

    if not ok:
        print("one or more simulation workers failed")
        return 1

    if args.update_command:
        if args.update_device is None:
            print("--update-command requires --update-device", file=sys.stderr)
            return 2
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.update_device
        print("\n=== Policy update command ===")
        print("physical GPU:", args.update_device)
        print("note: inside this process the update GPU is visible as cuda:0")
        print("command:", args.update_command)
        result = subprocess.run(args.update_command, shell=True, env=env)
        return result.returncode

    if args.update_device is not None:
        print(
            "no --update-command provided; coordinator only measured rollout workers. "
            "Use --update-command to launch the policy update on the update device."
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", nargs="?", default="mjmodel.xml")
    parser.add_argument("--impl", choices=("jax", "warp", "both"), default="warp")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--nconmax",
        type=int,
        default=64,
        help="MJWarp contacts per world. Ignored if --naconmax is set.",
    )
    parser.add_argument(
        "--naconmax",
        type=int,
        default=None,
        help="MJWarp total contacts across all worlds. Overrides --nconmax.",
    )
    parser.add_argument("--njmax", type=int, default=512, help="MJWarp constraints per world.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Root used to rewrite old G:\\DPPO\\... asset paths. Defaults to the repository root.",
    )
    parser.add_argument("--no-normalize-assets", action="store_true")
    parser.add_argument(
        "--sim-devices",
        default=None,
        help="Comma-separated physical GPU ids for rollout workers, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--update-device",
        default=None,
        help="Physical GPU id reserved for aggregation / policy update, e.g. 4.",
    )
    parser.add_argument(
        "--update-command",
        default=None,
        help="Optional shell command to run on --update-device after rollout workers finish.",
    )
    parser.add_argument(
        "--worker-index",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.sim_devices is not None:
        return _run_multi_gpu(args)

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        print(f"missing XML: {xml_path}", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2

    try:
        import jax
        import mujoco
    except Exception as exc:
        print(f"failed to import dependencies: {exc}", file=sys.stderr)
        print("Install with: pip install 'jax[cuda12]' 'mujoco-mjx[warp]'", file=sys.stderr)
        return 2

    print("xml:", xml_path)
    print("jax:", jax.__version__)
    print("jax backend:", jax.default_backend())
    print("jax devices:", jax.devices())
    print("mujoco:", mujoco.__version__)
    if args.worker_index is not None:
        print("worker index:", args.worker_index)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))

    try:
        if args.no_normalize_assets:
            mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
            changed = 0
        else:
            repo_root = (
                Path(args.repo_root).expanduser().resolve()
                if args.repo_root
                else Path(__file__).resolve().parents[1]
            )
            xml_string, changed = _rewrite_asset_paths(xml_path, repo_root)
            mj_model = mujoco.MjModel.from_xml_string(xml_string)
        if changed:
            print(f"rewritten asset paths: {changed}")
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
    except Exception as exc:
        print(f"MuJoCo failed to compile/forward the XML: {exc}", file=sys.stderr)
        return 1

    print("\n=== MuJoCo model summary ===")
    _print_model_summary(mj_model)

    impls = ("jax", "warp") if args.impl == "both" else (args.impl,)
    ok = True
    for impl in impls:
        try:
            _probe_impl(
                mj_model,
                impl,
                args.steps,
                args.batch_size,
                args.nconmax,
                args.naconmax,
                args.njmax,
            )
        except Exception as exc:
            ok = False
            print(f"\nMJX impl={impl} failed:")
            print(repr(exc))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
