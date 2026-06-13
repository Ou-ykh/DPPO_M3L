from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


RESET_TYPE_ALIASES = {
    "r": "reaching",
    "reaching": "reaching",
    "no": "near_object",
    "near_object": "near_object",
    "near-object": "near_object",
    "g": "stable_grasp",
    "grasp": "stable_grasp",
    "stable_grasp": "stable_grasp",
    "stable-grasp": "stable_grasp",
    "ng": "near_goal",
    "near_goal": "near_goal",
    "near-goal": "near_goal",
}
DEFAULT_RESET_TYPES = ("reaching", "near_object", "stable_grasp", "near_goal")
ARM_JOINT_NAMES = (
    "robot0_shoulder_pan_joint",
    "robot0_shoulder_lift_joint",
    "robot0_elbow_joint",
    "robot0_wrist_1_joint",
    "robot0_wrist_2_joint",
    "robot0_wrist_3_joint",
)
GRIPPER_JOINT_NAME = "right_driver_joint"
OBJECT_JOINT_NAME = "obj_jnt"
SITE_NAME = "center_point"
FINGER_SITE_NAMES = ("finger_left", "finger_right")
OBJECT_BODY_NAME = "object"
GOAL_BODY_NAME = "walls"
TABLE_GEOM_NAME = "work_tabletable_top"
DEFAULT_NOMINAL_ARM_QPOS = np.array(
    [-0.3, -0.6, 0.5, -1.57, -1.57, 1.57],
    dtype=np.float64,
)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
MIN_GRIPPER_DOWN_ALIGNMENT = 0.88
MIN_GRASP_OBJECT_HEIGHT_ABOVE_TABLE = 0.012
PAD_CONTACT_LOCAL_X_LIMIT = 0.0088
PAD_CONTACT_LOCAL_Z_MIN = 0.006
PAD_CONTACT_LOCAL_Z_MAX = 0.033
HELD_OBJECT_ROLLOUT_STEPS = 48
REACHING_GRIPPER_OBJECT_MIN = 0.115
REACHING_GRIPPER_OBJECT_MAX = 0.18
NEAR_OBJECT_GRIPPER_OBJECT_MIN = 0.045
NEAR_OBJECT_GRIPPER_OBJECT_MAX = 0.11
STABLE_GRASP_PRELOAD_CTRL_MIN = 0.38
STABLE_GRASP_PRELOAD_CTRL_MAX = 0.48
STABLE_GRASP_DIRECT_CLOSE_STEPS = 12
STABLE_GRASP_OBJECT_GOAL_MIN = 0.12
NEAR_GOAL_OBJECT_GOAL_MAX = 0.085
NEAR_GOAL_ARM_NOMINAL_DISTANCE_MAX = 1.15


def parse_reset_types(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_RESET_TYPES
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    if not parts:
        return DEFAULT_RESET_TYPES

    resolved: List[str] = []
    for part in parts:
        key = part.lower()
        if key in ("*", "all", "omni", "default"):
            for name in DEFAULT_RESET_TYPES:
                if name not in resolved:
                    resolved.append(name)
            continue
        if key not in RESET_TYPE_ALIASES:
            raise ValueError(
                f"unknown reset type '{part}'. valid values: "
                f"{sorted((*RESET_TYPE_ALIASES.keys(), 'all'))}"
            )
        name = RESET_TYPE_ALIASES[key]
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


@dataclass
class ResetState:
    reset_type: str
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    sample_index: int | None = None


class OmniResetSampler:
    """Generate OmniReset-style reset states and stabilized reset banks."""

    def __init__(
        self,
        mj_model,
        *,
        seed: int = 0,
        reset_types: str | Sequence[str] | None = None,
        stabilize_steps: int = 32,
        max_sample_attempts: int = 64,
        ik_max_iters: int = 80,
        ik_tolerance: float = 1e-3,
        arm_kp: float = 500.0,
        arm_kd: float = 80.0,
        nominal_arm_qpos: Sequence[float] | None = None,
        grasp_close_steps: int = 20,
        goal_offset_bank_size: int = 64,
        goal_offset_perturb_steps: int = 12,
        goal_offset_velocity_scale: float = 0.08,
    ):
        import mujoco

        self.mujoco = mujoco
        self.model = mj_model
        self.rng = np.random.default_rng(seed)
        self.reset_types = parse_reset_types(reset_types)
        self.stabilize_steps = int(stabilize_steps)
        self.max_sample_attempts = int(max_sample_attempts)
        self.ik_max_iters = int(ik_max_iters)
        self.ik_tolerance = float(ik_tolerance)
        self.arm_kp = float(arm_kp)
        self.arm_kd = float(arm_kd)
        self.grasp_close_steps = int(grasp_close_steps)
        self.goal_offset_bank_size = int(goal_offset_bank_size)
        self.goal_offset_perturb_steps = int(goal_offset_perturb_steps)
        self.goal_offset_velocity_scale = float(goal_offset_velocity_scale)
        self.nominal_arm_qpos = None if nominal_arm_qpos is None else np.asarray(
            nominal_arm_qpos, dtype=np.float64
        ).reshape(6)

        self.site_id = self._name_id(mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
        self.finger_site_ids = tuple(
            self._name_id(mujoco.mjtObj.mjOBJ_SITE, name) for name in FINGER_SITE_NAMES
        )
        self.object_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
        self.goal_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, GOAL_BODY_NAME)
        self.pad_body_ids = {
            "left_pad": self._name_id(mujoco.mjtObj.mjOBJ_BODY, "left_pad"),
            "right_pad": self._name_id(mujoco.mjtObj.mjOBJ_BODY, "right_pad"),
        }
        self.table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME
        )
        self.gripper_root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_mount"
        )
        self.robot_body_ids = {
            body_id
            for body_id in range(self.model.nbody)
            if self._is_robot_body(body_id)
        }

        self.arm_joint_ids = [
            self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES
        ]
        self.arm_qpos_ids = np.array(
            [int(self.model.jnt_qposadr[jid]) for jid in self.arm_joint_ids], dtype=np.int32
        )
        self.arm_dof_ids = np.array(
            [int(self.model.jnt_dofadr[jid]) for jid in self.arm_joint_ids], dtype=np.int32
        )
        self.arm_joint_ranges = np.asarray(
            [self.model.jnt_range[jid] for jid in self.arm_joint_ids], dtype=np.float64
        )

        self.gripper_joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT_NAME)
        self.gripper_qpos_id = int(self.model.jnt_qposadr[self.gripper_joint_id])
        self.object_joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT_NAME)
        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_dof_adr = int(self.model.jnt_dofadr[self.object_joint_id])

        ctrl_range = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        self.arm_ctrl_low = ctrl_range[:6, 0].copy()
        self.arm_ctrl_high = ctrl_range[:6, 1].copy()
        self.gripper_ctrl_open = float(ctrl_range[-1, 0])
        self.gripper_ctrl_closed = float(ctrl_range[-1, 1])
        self.gripper_ctrl_mid = 0.5 * (self.gripper_ctrl_open + self.gripper_ctrl_closed)

        self.base_data = self._new_data()
        self.base_object_pos = self.base_data.qpos[
            self.object_qpos_adr : self.object_qpos_adr + 3
        ].copy()
        self.base_object_quat = self.base_data.qpos[
            self.object_qpos_adr + 3 : self.object_qpos_adr + 7
        ].copy()
        self.grasp_object_geom_ids = self._grasp_object_geom_ids()
        self.goal_pos = self.base_data.xpos[self.goal_body_id].copy()
        self.default_arm_qpos = self.base_data.qpos[self.arm_qpos_ids].copy()
        if self.nominal_arm_qpos is None:
            self.nominal_arm_qpos = self.default_arm_qpos.copy()
        self.table_height = self._table_height(self.base_data)
        self.grasp_points_body = self._build_grasp_points_body()
        self.goal_offset_bank: np.ndarray | None = None

    def _name_id(self, obj_type, name: str) -> int:
        idx = self.mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"missing MuJoCo object '{name}'")
        return int(idx)

    def _body_has_ancestor(self, body_id: int, ancestor_id: int) -> bool:
        if ancestor_id < 0:
            return False
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return body_id == ancestor_id

    def _is_robot_body(self, body_id: int) -> bool:
        body_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or ""
        return body_name.startswith("robot0_") or self._body_has_ancestor(
            body_id, self.gripper_root_id
        )

    def _body_name(self, body_id: int) -> str:
        return (
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )

    def _geom_name(self, geom_id: int) -> str:
        return (
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            or ""
        )

    def _grasp_object_geom_ids(self) -> set[int]:
        geom_ids = np.flatnonzero(
            np.asarray(self.model.geom_bodyid, dtype=np.int32) == self.object_body_id
        )
        collision_ids = {
            int(geom_id)
            for geom_id in geom_ids.tolist()
            if self._geom_name(int(geom_id)).endswith("_collision")
        }
        if collision_ids:
            return collision_ids
        return {int(geom_id) for geom_id in geom_ids.tolist()}

    def _new_data(self):
        data = self.mujoco.MjData(self.model)
        if getattr(self.model, "nkey", 0) > 0:
            self.mujoco.mj_resetDataKeyframe(self.model, data, 0)
        else:
            self.mujoco.mj_resetData(self.model, data)
        self.mujoco.mj_forward(self.model, data)
        return data

    def _table_height(self, data) -> float:
        if self.table_geom_id >= 0:
            geom_pos = np.asarray(self.model.geom_pos[self.table_geom_id], dtype=np.float64)
            geom_size = np.asarray(self.model.geom_size[self.table_geom_id], dtype=np.float64)
            return float(geom_pos[2] + geom_size[2])
        return float(data.qpos[self.object_qpos_adr + 2])

    def _quat_from_yaw(self, yaw: float) -> np.ndarray:
        return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float64)

    def _quat_to_rotmat(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(value) for value in quat]
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _body_to_world(self, body_point: np.ndarray, body_pos: np.ndarray, body_quat: np.ndarray) -> np.ndarray:
        return body_pos + self._quat_to_rotmat(body_quat) @ np.asarray(body_point, dtype=np.float64)

    def _build_grasp_points_body(self) -> np.ndarray:
        object_geom_ids = np.flatnonzero(
            np.asarray(self.model.geom_bodyid, dtype=np.int32) == self.object_body_id
        )
        collision_geom_ids = [
            int(geom_id)
            for geom_id in object_geom_ids.tolist()
            if self._geom_name(int(geom_id)).endswith("_collision")
        ]
        holder_geom_ids = [
            geom_id
            for geom_id in collision_geom_ids
            if "holder" in self._geom_name(geom_id).lower()
        ]
        geom_ids = holder_geom_ids or collision_geom_ids or [
            int(geom_id) for geom_id in object_geom_ids.tolist()
        ]
        if not geom_ids:
            return np.zeros((1, 3), dtype=np.float64)

        mins = []
        maxs = []
        for geom_id in geom_ids:
            center = np.asarray(self.model.geom_pos[geom_id], dtype=np.float64)
            size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64).copy()
            if np.isfinite(size).all() and float(np.max(size)) > 1e-5:
                extent = np.maximum(size, 0.004)
                mins.append(center - extent)
                maxs.append(center + extent)
            else:
                radius = max(float(self.model.geom_rbound[geom_id]), 0.004)
                mins.append(center - radius)
                maxs.append(center + radius)

        bbox_min = np.min(np.stack(mins, axis=0), axis=0)
        bbox_max = np.max(np.stack(maxs, axis=0), axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        extent = np.maximum(0.5 * (bbox_max - bbox_min), np.array([0.008, 0.008, 0.008], dtype=np.float64))

        # These are center-of-grasp targets, not surface contact points. Keeping
        # them near the holder/collision center prevents reset fallback from
        # hanging the object on a fingertip or pad edge.
        points = [
            center,
            center + np.array([+0.18 * extent[0], 0.0, 0.0], dtype=np.float64),
            center + np.array([-0.18 * extent[0], 0.0, 0.0], dtype=np.float64),
            center + np.array([0.0, +0.18 * extent[1], 0.0], dtype=np.float64),
            center + np.array([0.0, -0.18 * extent[1], 0.0], dtype=np.float64),
            center + np.array([0.0, 0.0, +0.18 * extent[2]], dtype=np.float64),
            center + np.array([0.0, 0.0, -0.18 * extent[2]], dtype=np.float64),
        ]
        return np.stack(points, axis=0)

    def _sample_grasp_point_body(self) -> np.ndarray:
        return self.grasp_points_body[
            int(self.rng.integers(0, len(self.grasp_points_body)))
        ].copy()

    def _sample_grasp_point_world(
        self,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        grasp_point_body: np.ndarray | None = None,
    ) -> np.ndarray:
        if grasp_point_body is None:
            grasp_point_body = self._sample_grasp_point_body()
        return self._body_to_world(grasp_point_body, obj_pos, obj_quat)

    def _model_signature(self) -> str:
        geom_mask = np.asarray(self.model.geom_bodyid, dtype=np.int32) == self.object_body_id
        geom_pos = np.asarray(self.model.geom_pos[geom_mask], dtype=np.float64).reshape(-1)
        geom_rbound = np.asarray(self.model.geom_rbound[geom_mask], dtype=np.float64).reshape(-1)
        joint_dynamics = np.concatenate(
            (
                np.asarray(self.model.jnt_range, dtype=np.float64).reshape(-1),
                np.asarray(self.model.jnt_stiffness, dtype=np.float64).reshape(-1),
                np.asarray(self.model.dof_damping, dtype=np.float64).reshape(-1),
                np.asarray(self.model.dof_frictionloss, dtype=np.float64).reshape(-1),
                np.asarray(self.model.dof_armature, dtype=np.float64).reshape(-1),
            ),
            axis=0,
        )
        actuator_dynamics = np.concatenate(
            (
                np.asarray(self.model.actuator_ctrlrange, dtype=np.float64).reshape(-1),
                np.asarray(self.model.actuator_forcerange, dtype=np.float64).reshape(-1),
                np.asarray(self.model.actuator_gainprm, dtype=np.float64).reshape(-1),
                np.asarray(self.model.actuator_biasprm, dtype=np.float64).reshape(-1),
            ),
            axis=0,
        )
        payload = np.concatenate(
            (
                np.array(
                    [self.model.nq, self.model.nv, self.model.nu, self.model.ngeom, self.model.nmesh],
                    dtype=np.float64,
                ),
                geom_pos,
                geom_rbound,
                self.goal_pos.astype(np.float64, copy=False).reshape(-1),
                joint_dynamics,
                actuator_dynamics,
            ),
            axis=0,
        )
        return hashlib.sha1(payload.tobytes()).hexdigest()[:16]

    def _set_arm_qpos(self, data, qpos: np.ndarray) -> None:
        data.qpos[self.arm_qpos_ids] = qpos
        data.qvel[self.arm_dof_ids] = 0.0

    def _set_object_pose(self, data, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3] = pos
        data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = (
            self.base_object_quat if quat is None else quat
        )
        data.qvel[self.object_dof_adr : self.object_dof_adr + 6] = 0.0

    def _metrics(self, data) -> Dict[str, float]:
        obj_pos = data.xpos[self.object_body_id].copy()
        obj_qpos = data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3].copy()
        goal_pos = data.xpos[self.goal_body_id].copy()
        site_pos = data.site_xpos[self.site_id].copy()
        finger_left = data.site_xpos[self.finger_site_ids[0]].copy()
        finger_right = data.site_xpos[self.finger_site_ids[1]].copy()
        gripper_center = 0.5 * (finger_left + finger_right)
        arm_qpos = data.qpos[self.arm_qpos_ids].copy()
        site_rot = data.site_xmat[self.site_id].reshape(3, 3)
        return {
            "object_goal": float(np.linalg.norm(obj_pos - goal_pos)),
            "object_qpos_goal": float(np.linalg.norm(obj_qpos - goal_pos)),
            "site_object": float(np.linalg.norm(site_pos - obj_pos)),
            "gripper_object": float(np.linalg.norm(gripper_center - obj_pos)),
            "site_z": float(site_pos[2]),
            "gripper_down_alignment": float(np.dot(site_rot[:, 0], WORLD_UP)),
            "arm_nominal_distance": float(np.linalg.norm(arm_qpos - self.nominal_arm_qpos)),
            "object_z": float(obj_pos[2]),
            "max_qvel": float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0,
        }

    def _gripper_center(self, data) -> np.ndarray:
        finger_left = data.site_xpos[self.finger_site_ids[0]].copy()
        finger_right = data.site_xpos[self.finger_site_ids[1]].copy()
        return 0.5 * (finger_left + finger_right)

    def _finalize_state(self, data, reset_type: str, gripper_ctrl: float) -> ResetState | None:
        ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()
        ctrl[:6] = 0.0
        ctrl[6] = gripper_ctrl
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = ctrl
        self.mujoco.mj_forward(self.model, data)
        metrics = self._metrics(data)
        if not self._valid_for_type(reset_type, data, metrics):
            return None
        if reset_type == "stable_grasp":
            target_arm_qpos = data.qpos[self.arm_qpos_ids].copy()
            for _ in range(HELD_OBJECT_ROLLOUT_STEPS):
                self._apply_hold_control(data, target_arm_qpos, gripper_ctrl)
                self.mujoco.mj_step(self.model, data)
            metrics = self._metrics(data)
            if not self._valid_for_type(reset_type, data, metrics):
                return None

            ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()
            ctrl[:6] = 0.0
            ctrl[6] = gripper_ctrl
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[:] = ctrl
            self.mujoco.mj_forward(self.model, data)
            metrics = self._metrics(data)
            if not self._valid_for_type(reset_type, data, metrics):
                return None
        return ResetState(
            reset_type=reset_type,
            qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel=np.zeros_like(np.asarray(data.qvel, dtype=np.float64)),
            ctrl=ctrl,
        )

    def _apply_hold_control(self, data, target_arm_qpos: np.ndarray, gripper_ctrl: float) -> None:
        q_current = data.qpos[self.arm_qpos_ids]
        q_vel = data.qvel[self.arm_dof_ids]
        gravity = data.qfrc_bias[self.arm_dof_ids] - data.qfrc_passive[self.arm_dof_ids]
        tau = self.arm_kp * (target_arm_qpos - q_current) - self.arm_kd * q_vel + gravity
        data.ctrl[:6] = np.clip(tau, self.arm_ctrl_low, self.arm_ctrl_high)
        data.ctrl[6] = gripper_ctrl

    def _stabilize(self, data, target_arm_qpos: np.ndarray, gripper_ctrl: float) -> None:
        for _ in range(self.stabilize_steps):
            self._apply_hold_control(data, target_arm_qpos, gripper_ctrl)
            self.mujoco.mj_step(self.model, data)

    def _solve_ik_to_position(
        self,
        data,
        target_pos: np.ndarray,
        initial_qpos: np.ndarray | None = None,
    ) -> np.ndarray | None:
        qpos = self.nominal_arm_qpos.copy() if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float64).copy()
        self._set_arm_qpos(data, qpos)
        data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, data)

        for _ in range(self.ik_max_iters):
            site_pos = data.site_xpos[self.site_id].copy()
            error = target_pos - site_pos
            site_rot = data.site_xmat[self.site_id].reshape(3, 3)
            tool_up_axis = site_rot[:, 0]
            orientation_error = np.cross(tool_up_axis, WORLD_UP)
            if np.linalg.norm(error) < self.ik_tolerance:
                return data.qpos[self.arm_qpos_ids].copy()

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            self.mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
            orientation_weight = 0.25
            jac = np.vstack(
                (
                    jacp[:, self.arm_dof_ids],
                    orientation_weight * jacr[:, self.arm_dof_ids],
                )
            )
            task_error = np.concatenate((error, orientation_weight * orientation_error), axis=0)
            jac_pinv = np.linalg.pinv(jac, rcond=1e-3)
            nullspace = np.eye(len(self.arm_dof_ids), dtype=np.float64) - jac_pinv @ jac
            posture_delta = 0.08 * (self.nominal_arm_qpos - data.qpos[self.arm_qpos_ids])
            delta = jac_pinv @ task_error + nullspace @ posture_delta
            delta = np.clip(delta, -0.05, 0.05)
            qpos = np.clip(
                data.qpos[self.arm_qpos_ids] + 0.8 * delta,
                self.arm_joint_ranges[:, 0],
                self.arm_joint_ranges[:, 1],
            )
            self._set_arm_qpos(data, qpos)
            self.mujoco.mj_forward(self.model, data)
        return None

    def _sample_table_object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.base_object_pos.copy()
        pos[0] += self.rng.uniform(-0.05, 0.05)
        pos[1] += self.rng.uniform(-0.05, 0.05)
        pos[2] = self.table_height
        quat = self._quat_from_yaw(float(self.rng.uniform(-np.pi, np.pi)))
        return pos, quat

    def _sample_target_near_object(
        self,
        obj_pos: np.ndarray,
        *,
        lateral_radius: float,
        min_height: float,
        max_height: float,
    ) -> np.ndarray:
        angle = float(self.rng.uniform(-np.pi, np.pi))
        radius = float(self.rng.uniform(0.0, lateral_radius))
        return obj_pos + np.array(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                self.rng.uniform(min_height, max_height),
            ],
            dtype=np.float64,
        )

    def _close_gripper(
        self,
        data,
        target_arm_qpos: np.ndarray,
        start_ctrl: float,
        end_ctrl: float,
        steps: int,
    ) -> None:
        for step in range(max(int(steps), 1)):
            alpha = float(step + 1) / float(max(int(steps), 1))
            ctrl = (1.0 - alpha) * start_ctrl + alpha * end_ctrl
            self._apply_hold_control(data, target_arm_qpos, ctrl)
            self.mujoco.mj_step(self.model, data)

    def _build_goal_offset_bank(self, bank_size: int) -> np.ndarray:
        offsets: List[np.ndarray] = []
        attempts = 0
        max_attempts = max(bank_size * 12, 64)
        while len(offsets) < bank_size and attempts < max_attempts:
            attempts += 1
            data = self._new_data()
            self._set_arm_qpos(data, self.default_arm_qpos)
            self._set_object_pose(data, self.goal_pos, self.base_object_quat.copy())
            data.ctrl[:] = 0.0
            self.mujoco.mj_forward(self.model, data)

            data.qvel[self.object_dof_adr : self.object_dof_adr + 3] = self.rng.uniform(
                -self.goal_offset_velocity_scale,
                self.goal_offset_velocity_scale,
                size=3,
            )
            data.qvel[self.object_dof_adr + 3 : self.object_dof_adr + 6] = self.rng.uniform(
                -0.5 * self.goal_offset_velocity_scale,
                0.5 * self.goal_offset_velocity_scale,
                size=3,
            )
            for _ in range(self.goal_offset_perturb_steps):
                self.mujoco.mj_step(self.model, data)

            offset_pos = data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3] - self.goal_pos
            distance = float(np.linalg.norm(offset_pos))
            metrics = self._metrics(data)
            if not self._common_valid(data, metrics):
                continue
            if not (0.006 <= distance <= 0.04):
                continue
            offsets.append(
                np.concatenate(
                    (
                        offset_pos.astype(np.float64, copy=False),
                        data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7].copy(),
                    ),
                    axis=0,
                )
            )

        if not offsets:
            fallback = np.concatenate(
                (np.array([0.012, 0.0, 0.0], dtype=np.float64), self.base_object_quat.copy()),
                axis=0,
            )
            return np.repeat(fallback[None, :], max(bank_size, 1), axis=0)
        if len(offsets) < bank_size:
            indices = self.rng.integers(0, len(offsets), size=bank_size - len(offsets))
            offsets.extend([offsets[int(index)].copy() for index in indices])
        return np.stack(offsets[:bank_size], axis=0)

    def _common_valid(self, data, metrics: Dict[str, float]) -> bool:
        if not np.isfinite(data.qpos).all():
            return False
        if not np.isfinite(data.qvel).all():
            return False
        if metrics["object_z"] < self.table_height - 0.02:
            return False
        if metrics["max_qvel"] > 25.0:
            return False
        if metrics["gripper_down_alignment"] < MIN_GRIPPER_DOWN_ALIGNMENT:
            return False
        return True

    def _contact_valid(self, reset_type: str, data, penetration_tol: float = 1e-4) -> bool:
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if float(contact.dist) >= -penetration_tol:
                continue

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            object_goal_pair = (
                {body1, body2} == {self.object_body_id, self.goal_body_id}
            )
            if object_goal_pair:
                return False
            robot1 = body1 in self.robot_body_ids
            robot2 = body2 in self.robot_body_ids
            if robot1 == robot2:
                continue

            robot_body = body1 if robot1 else body2
            other_body = body2 if robot1 else body1
            other_geom = geom2 if robot1 else geom1

            other_is_object = other_body == self.object_body_id
            other_is_goal = other_body == self.goal_body_id
            other_is_table = other_geom == self.table_geom_id
            if not (other_is_object or other_is_goal or other_is_table):
                continue

            robot_body_name = self._body_name(robot_body)
            if reset_type in ("stable_grasp", "near_goal") and other_is_object:
                if robot_body_name in ("left_pad", "right_pad"):
                    continue
            return False
        return True

    def _pad_object_contact_names(self, data, contact_tol: float = 1e-4) -> set[str]:
        pad_names: set[str] = set()
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if float(contact.dist) > contact_tol:
                continue

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if self.object_body_id not in (body1, body2):
                continue

            other_body = body2 if body1 == self.object_body_id else body1
            object_geom = geom1 if body1 == self.object_body_id else geom2
            if object_geom not in self.grasp_object_geom_ids:
                continue
            body_name = self._body_name(other_body)
            if body_name in ("left_pad", "right_pad"):
                pad_names.add(body_name)
        return pad_names

    def _has_two_sided_pad_object_contact(self, data, contact_tol: float = 1e-4) -> bool:
        return {"left_pad", "right_pad"}.issubset(
            self._pad_object_contact_names(data, contact_tol=contact_tol)
        )

    def _pad_object_contact_local_positions(
        self,
        data,
        contact_tol: float = 1e-4,
    ) -> Dict[str, List[np.ndarray]]:
        contacts: Dict[str, List[np.ndarray]] = {"left_pad": [], "right_pad": []}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if float(contact.dist) > contact_tol:
                continue

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if self.object_body_id not in (body1, body2):
                continue

            object_geom = geom1 if body1 == self.object_body_id else geom2
            if object_geom not in self.grasp_object_geom_ids:
                continue

            pad_body = body2 if body1 == self.object_body_id else body1
            pad_name = self._body_name(pad_body)
            if pad_name not in contacts:
                continue

            pad_rot = data.xmat[pad_body].reshape(3, 3)
            pad_pos = data.xpos[pad_body]
            local_pos = pad_rot.T @ (np.asarray(contact.pos, dtype=np.float64) - pad_pos)
            contacts[pad_name].append(local_pos)
        return contacts

    def _has_centered_pad_object_contact(self, data) -> bool:
        contacts = self._pad_object_contact_local_positions(data)
        for pad_name in ("left_pad", "right_pad"):
            local_positions = contacts[pad_name]
            if not local_positions:
                return False
            local = np.stack(local_positions, axis=0)
            centered = (
                (np.abs(local[:, 0]) <= PAD_CONTACT_LOCAL_X_LIMIT)
                & (local[:, 2] >= PAD_CONTACT_LOCAL_Z_MIN)
                & (local[:, 2] <= PAD_CONTACT_LOCAL_Z_MAX)
            )
            if not bool(np.any(centered)):
                return False
        return True

    def _has_object_table_contact(self, data, contact_tol: float = 1e-4) -> bool:
        if self.table_geom_id < 0:
            return False
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if float(contact.dist) > contact_tol:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if self.object_body_id not in (body1, body2):
                continue
            other_geom = geom2 if body1 == self.object_body_id else geom1
            if other_geom == self.table_geom_id:
                return True
        return False

    def _held_object_valid(self, data, metrics: Dict[str, float]) -> bool:
        return (
            metrics["object_z"] >= self.table_height + MIN_GRASP_OBJECT_HEIGHT_ABOVE_TABLE
            and not self._has_object_table_contact(data)
            and self._has_two_sided_pad_object_contact(data)
            and self._has_centered_pad_object_contact(data)
        )

    def _valid_for_type(self, reset_type: str, data, metrics: Dict[str, float]) -> bool:
        if not self._common_valid(data, metrics):
            return False
        if not self._contact_valid(reset_type, data):
            return False

        object_goal = metrics["object_goal"]
        object_qpos_goal = metrics["object_qpos_goal"]
        gripper_object = metrics["gripper_object"]
        object_z = metrics["object_z"]
        site_z = metrics["site_z"]
        arm_nominal_distance = metrics["arm_nominal_distance"]

        arm_nominal_distance_limit = (
            NEAR_GOAL_ARM_NOMINAL_DISTANCE_MAX
            if reset_type == "near_goal"
            else 0.65
        )
        if arm_nominal_distance > arm_nominal_distance_limit:
            return False
        if reset_type == "reaching" and site_z < self.table_height + 0.07:
            return False
        if reset_type == "near_object" and site_z < self.table_height + 0.075:
            return False
        if reset_type in ("stable_grasp", "near_goal") and site_z < self.table_height + 0.025:
            return False

        if reset_type == "reaching":
            return (
                REACHING_GRIPPER_OBJECT_MIN <= gripper_object <= REACHING_GRIPPER_OBJECT_MAX
                and object_goal > 0.03
            )
        if reset_type == "near_object":
            return (
                NEAR_OBJECT_GRIPPER_OBJECT_MIN <= gripper_object <= NEAR_OBJECT_GRIPPER_OBJECT_MAX
                and object_z <= self.table_height + 0.04
            )
        if reset_type == "stable_grasp":
            return (
                gripper_object <= 0.05
                and metrics["site_object"] <= 0.06
                and object_goal >= STABLE_GRASP_OBJECT_GOAL_MIN
                and float(data.ctrl[6]) >= self.gripper_ctrl_mid
                and self._held_object_valid(data, metrics)
            )
        if reset_type == "near_goal":
            return (
                0.012 <= object_qpos_goal <= 0.075
                and object_goal <= NEAR_GOAL_OBJECT_GOAL_MAX
                and metrics["site_object"] <= 0.06
                and gripper_object <= 0.05
                and float(data.ctrl[6]) >= self.gripper_ctrl_mid
                and self._held_object_valid(data, metrics)
            )
        return False

    def _make_candidate(self, reset_type: str) -> ResetState | None:
        data = self._new_data()

        if reset_type == "reaching":
            obj_pos, obj_quat = self._sample_table_object_pose()
            grasp_point = self._sample_grasp_point_world(obj_pos, obj_quat)
            angle = float(self.rng.uniform(-np.pi, np.pi))
            lateral_radius = float(self.rng.uniform(0.055, 0.105))
            target = grasp_point + np.array(
                [
                    lateral_radius * np.cos(angle),
                    lateral_radius * np.sin(angle),
                    self.rng.uniform(0.075, 0.125),
                ],
                dtype=np.float64,
            )
            gripper_ctrl = self.gripper_ctrl_open
        elif reset_type == "near_object":
            obj_pos, obj_quat = self._sample_table_object_pose()
            grasp_point = self._sample_grasp_point_world(obj_pos, obj_quat)
            target = grasp_point + np.array(
                [
                    self.rng.uniform(-0.02, 0.02),
                    self.rng.uniform(-0.02, 0.02),
                    self.rng.uniform(0.05, 0.08),
                ],
                dtype=np.float64,
            )
            gripper_ctrl = float(
                self.rng.choice(
                    np.array(
                        [self.gripper_ctrl_open, self.gripper_ctrl_mid],
                        dtype=np.float64,
                    )
                )
            )
        elif reset_type == "stable_grasp":
            obj_pos, obj_quat = self._sample_table_object_pose()
            grasp_point_body = self.grasp_points_body[0].copy()
            grasp_point = self._sample_grasp_point_world(obj_pos, obj_quat, grasp_point_body)
            pregrasp_target = grasp_point + np.array(
                [
                    self.rng.uniform(-0.010, 0.010),
                    self.rng.uniform(-0.010, 0.010),
                    self.rng.uniform(0.035, 0.055),
                ],
                dtype=np.float64,
            )
            arm_qpos = self._solve_ik_to_position(data, pregrasp_target)
            if arm_qpos is None:
                return None
            self._set_arm_qpos(data, arm_qpos)
            data.ctrl[:] = 0.0
            self.mujoco.mj_forward(self.model, data)

            preload_ctrl = float(
                self.rng.uniform(STABLE_GRASP_PRELOAD_CTRL_MIN, STABLE_GRASP_PRELOAD_CTRL_MAX)
            )
            for _ in range(5):
                self._stabilize(data, arm_qpos, preload_ctrl)

            gripper_center = self._gripper_center(data)
            obj_pos = gripper_center - self._quat_to_rotmat(obj_quat) @ grasp_point_body
            obj_pos = obj_pos + np.array(
                [
                    self.rng.uniform(-0.0005, 0.0005),
                    self.rng.uniform(-0.0005, 0.0005),
                    self.rng.uniform(-0.0005, 0.0005),
                ],
                dtype=np.float64,
            )
            obj_pos[2] = max(obj_pos[2], self.table_height + self.rng.uniform(0.018, 0.032))
            self._set_object_pose(data, obj_pos, obj_quat)
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[6] = preload_ctrl
            self.mujoco.mj_forward(self.model, data)

            self._close_gripper(
                data,
                arm_qpos,
                preload_ctrl,
                self.gripper_ctrl_closed,
                STABLE_GRASP_DIRECT_CLOSE_STEPS,
            )
            for _ in range(2):
                self._stabilize(data, arm_qpos, self.gripper_ctrl_closed)

            metrics = self._metrics(data)
            if self._valid_for_type(reset_type, data, metrics):
                return self._finalize_state(data, reset_type, self.gripper_ctrl_closed)
            return None
        elif reset_type == "near_goal":
            angle = float(self.rng.uniform(-np.pi, np.pi))
            radius = float(self.rng.uniform(0.022, 0.050))
            obj_pos = self.goal_pos + np.array(
                [radius * np.cos(angle), radius * np.sin(angle), 0.0],
                dtype=np.float64,
            )
            obj_pos[2] = self.table_height
            obj_quat = self._quat_from_yaw(float(self.rng.uniform(-np.pi, np.pi)))
            grasp_point_body = self.grasp_points_body[0].copy()
            grasp_point = self._sample_grasp_point_world(obj_pos, obj_quat, grasp_point_body)
            target = grasp_point + np.array(
                [
                    self.rng.uniform(-0.008, 0.008),
                    self.rng.uniform(-0.008, 0.008),
                    self.rng.uniform(0.040, 0.060),
                ],
                dtype=np.float64,
            )
            arm_qpos = self._solve_ik_to_position(data, target)
            if arm_qpos is None:
                return None
            self._set_arm_qpos(data, arm_qpos)
            data.ctrl[:] = 0.0
            self.mujoco.mj_forward(self.model, data)

            preload_ctrl = float(
                self.rng.uniform(STABLE_GRASP_PRELOAD_CTRL_MIN, STABLE_GRASP_PRELOAD_CTRL_MAX)
            )
            for _ in range(5):
                self._stabilize(data, arm_qpos, preload_ctrl)

            gripper_center = self._gripper_center(data)
            obj_pos = gripper_center - self._quat_to_rotmat(obj_quat) @ grasp_point_body
            goal_offset = obj_pos[:2] - self.goal_pos[:2]
            goal_distance_xy = float(np.linalg.norm(goal_offset))
            if goal_distance_xy > 0.060:
                obj_pos[:2] = self.goal_pos[:2] + goal_offset / max(goal_distance_xy, 1e-6) * 0.052
            elif goal_distance_xy < 0.018:
                direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
                obj_pos[:2] = self.goal_pos[:2] + 0.022 * direction
            obj_pos += np.array(
                [
                    self.rng.uniform(-0.001, 0.001),
                    self.rng.uniform(-0.001, 0.001),
                    self.rng.uniform(-0.0005, 0.0005),
                ],
                dtype=np.float64,
            )
            obj_pos[2] = max(obj_pos[2], self.table_height + self.rng.uniform(0.014, 0.030))
            self._set_object_pose(data, obj_pos, obj_quat)
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[6] = preload_ctrl
            self.mujoco.mj_forward(self.model, data)
            self._close_gripper(
                data,
                arm_qpos,
                preload_ctrl,
                self.gripper_ctrl_closed,
                STABLE_GRASP_DIRECT_CLOSE_STEPS,
            )
            for _ in range(2):
                self._stabilize(data, arm_qpos, self.gripper_ctrl_closed)

            metrics = self._metrics(data)
            if not self._valid_for_type(reset_type, data, metrics):
                return None
            return self._finalize_state(data, reset_type, self.gripper_ctrl_closed)
        else:
            raise ValueError(f"unsupported reset type '{reset_type}'")

        arm_qpos = self._solve_ik_to_position(data, target)
        if arm_qpos is None:
            return None

        self._set_arm_qpos(data, arm_qpos)
        self._set_object_pose(data, obj_pos, obj_quat)
        data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, data)
        self._stabilize(data, arm_qpos, gripper_ctrl)
        metrics = self._metrics(data)
        if not self._valid_for_type(reset_type, data, metrics):
            return None

        return self._finalize_state(data, reset_type, gripper_ctrl)

    def sample(self, reset_type: str) -> ResetState | None:
        canonical = parse_reset_types([reset_type])[0]
        for _ in range(self.max_sample_attempts):
            sample = self._make_candidate(canonical)
            if sample is not None:
                return sample
        return None

    def sample_uniform(self) -> ResetState:
        for _ in range(self.max_sample_attempts * max(len(self.reset_types), 1)):
            reset_type = self.reset_types[int(self.rng.integers(0, len(self.reset_types)))]
            sample = self.sample(reset_type)
            if sample is not None:
                return sample
        raise RuntimeError("failed to sample a valid OmniReset state")

    def _cache_metadata(self, bank_size_per_type: int) -> Dict[str, object]:
        return {
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "model_signature": self._model_signature(),
            "reset_types": list(self.reset_types),
            "bank_size_per_type": int(bank_size_per_type),
            "stabilize_steps": int(self.stabilize_steps),
            "grasp_close_steps": int(self.grasp_close_steps),
            "goal_offset_bank_size": int(self.goal_offset_bank_size),
            "goal_offset_perturb_steps": int(self.goal_offset_perturb_steps),
            "goal_offset_velocity_scale": float(self.goal_offset_velocity_scale),
            "zero_reset_qvel": True,
            "near_goal_qpos_metric": True,
            "near_goal_grasp_required": "finalized_lifted_centered_dynamic_closed_gripper_v7",
            "stable_grasp_pad_contact_required": "direct_preloaded_centered_dynamic_pad_contact_v6",
            "reset_pose_height_guard": "pregrasp_hover_v2",
            "stable_grasp_site_height_guard": "table_plus_0p025",
            "gripper_down_alignment": float(MIN_GRIPPER_DOWN_ALIGNMENT),
            "min_grasp_object_height_above_table": float(MIN_GRASP_OBJECT_HEIGHT_ABOVE_TABLE),
            "finalized_state_validation": True,
            "phase_distance_bands": {
                "reaching_gripper_object": [
                    float(REACHING_GRIPPER_OBJECT_MIN),
                    float(REACHING_GRIPPER_OBJECT_MAX),
                ],
                "near_object_gripper_object": [
                    float(NEAR_OBJECT_GRIPPER_OBJECT_MIN),
                    float(NEAR_OBJECT_GRIPPER_OBJECT_MAX),
                ],
                "stable_grasp_object_goal_min": float(STABLE_GRASP_OBJECT_GOAL_MIN),
                "near_goal_object_goal_max": float(NEAR_GOAL_OBJECT_GOAL_MAX),
                "reaching_target_sampler": "mid_lateral_hover_v3",
            },
            "arm_nominal_distance_limits": {
                "default": 0.65,
                "near_goal": float(NEAR_GOAL_ARM_NOMINAL_DISTANCE_MAX),
            },
            "stable_grasp_sampling": {
                "mode": "direct_preloaded_holder_center_v2",
                "preload_ctrl_range": [
                    float(STABLE_GRASP_PRELOAD_CTRL_MIN),
                    float(STABLE_GRASP_PRELOAD_CTRL_MAX),
                ],
                "direct_close_steps": int(STABLE_GRASP_DIRECT_CLOSE_STEPS),
            },
            "centered_pad_contact": {
                "local_x_abs_max": float(PAD_CONTACT_LOCAL_X_LIMIT),
                "local_z_min": float(PAD_CONTACT_LOCAL_Z_MIN),
                "local_z_max": float(PAD_CONTACT_LOCAL_Z_MAX),
                "object_geoms": "collision_only",
            },
            "grasp_points_body": "holder_or_collision_center_v2",
            "near_goal_sampling": "direct_preloaded_centered_goal_v3",
            "held_object_rollout_steps": int(HELD_OBJECT_ROLLOUT_STEPS),
            "nominal_arm_qpos": [float(value) for value in self.nominal_arm_qpos],
        }

    def _cache_metadata_is_compatible(
        self,
        metadata: Dict[str, object],
        bank_size_per_type: int,
    ) -> bool:
        try:
            stored_bank_size = int(metadata.get("bank_size_per_type", -1))
        except (TypeError, ValueError):
            return False
        if stored_bank_size != int(bank_size_per_type):
            return False

        stored_reset_types = set(str(value) for value in metadata.get("reset_types", []))
        if not set(self.reset_types).issubset(stored_reset_types):
            return False

        expected = self._cache_metadata(stored_bank_size)
        for key in (
            "nq",
            "nv",
            "nu",
            "model_signature",
            "stabilize_steps",
            "grasp_close_steps",
            "goal_offset_bank_size",
            "goal_offset_perturb_steps",
            "goal_offset_velocity_scale",
            "zero_reset_qvel",
            "near_goal_qpos_metric",
            "near_goal_grasp_required",
            "stable_grasp_pad_contact_required",
            "reset_pose_height_guard",
            "stable_grasp_site_height_guard",
            "gripper_down_alignment",
            "min_grasp_object_height_above_table",
            "finalized_state_validation",
            "phase_distance_bands",
            "arm_nominal_distance_limits",
            "stable_grasp_sampling",
            "centered_pad_contact",
            "grasp_points_body",
            "near_goal_sampling",
            "held_object_rollout_steps",
            "nominal_arm_qpos",
        ):
            if metadata.get(key) != expected.get(key):
                return False
        return True

    def _resolve_cache_path(
        self,
        cache_path: str | Path | None,
        bank_size_per_type: int,
    ) -> Path | None:
        if cache_path is None:
            return None
        path = Path(cache_path)
        if path.suffix.lower() == ".npz":
            return path
        metadata = self._cache_metadata(bank_size_per_type)
        signature = hashlib.sha1(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return path / f"omnireset_bank_{signature}.npz"

    def _find_compatible_cache_path(
        self,
        cache_path: str | Path | None,
        bank_size_per_type: int,
    ) -> Path | None:
        if cache_path is None:
            return None
        path = Path(cache_path)
        if path.suffix.lower() == ".npz" or not path.exists():
            return None

        candidates = sorted(
            path.glob("omnireset_bank_*.npz"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                with np.load(candidate, allow_pickle=False) as data:
                    metadata = json.loads(str(data["__metadata_json"]))
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
            if self._cache_metadata_is_compatible(metadata, bank_size_per_type):
                return candidate
        return None

    def _save_bank(self, path: Path, bank: Dict[str, Dict[str, np.ndarray]], bank_size_per_type: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, np.ndarray] = {
            "__metadata_json": np.asarray(json.dumps(self._cache_metadata(bank_size_per_type)))
        }
        for reset_type, entry in bank.items():
            for key, value in entry.items():
                payload[f"{reset_type}__{key}"] = np.asarray(value, dtype=np.float32)
        np.savez_compressed(path, **payload)

    def _load_bank(self, path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["__metadata_json"]))
            if not self._cache_metadata_is_compatible(
                metadata,
                int(metadata["bank_size_per_type"]),
            ):
                raise ValueError("incompatible reset bank cache metadata")
            expected = self._cache_metadata(int(metadata["bank_size_per_type"]))
            for key in (
                "nq",
                "nv",
                "nu",
                "model_signature",
                "stabilize_steps",
                "grasp_close_steps",
                "goal_offset_bank_size",
                "goal_offset_perturb_steps",
                "goal_offset_velocity_scale",
                "zero_reset_qvel",
                "near_goal_qpos_metric",
                "near_goal_grasp_required",
                "stable_grasp_pad_contact_required",
                "reset_pose_height_guard",
                "stable_grasp_site_height_guard",
                "gripper_down_alignment",
                "min_grasp_object_height_above_table",
                "finalized_state_validation",
                "phase_distance_bands",
                "arm_nominal_distance_limits",
                "stable_grasp_sampling",
                "centered_pad_contact",
                "grasp_points_body",
                "near_goal_sampling",
                "held_object_rollout_steps",
                "nominal_arm_qpos",
            ):
                if metadata.get(key) != expected.get(key):
                    raise ValueError(f"incompatible reset bank cache field '{key}'")
            bank: Dict[str, Dict[str, np.ndarray]] = {}
            for reset_type in self.reset_types:
                entry: Dict[str, np.ndarray] = {}
                for key in (
                    "qpos",
                    "qvel",
                    "ctrl",
                    "arm_qpos",
                    "gripper_qpos",
                    "gripper_ctrl",
                ):
                    payload_key = f"{reset_type}__{key}"
                    if payload_key in data:
                        entry[key] = np.asarray(data[payload_key], dtype=np.float32)
                bank[reset_type] = entry
        return bank

    def build_bank(
        self,
        bank_size_per_type: int,
        cache_path: str | Path | None = None,
        force_rebuild: bool = False,
        verbose: bool = False,
        log_prefix: str = "",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        resolved_cache_path = self._resolve_cache_path(cache_path, bank_size_per_type)
        if resolved_cache_path is not None and resolved_cache_path.exists() and not force_rebuild:
            if verbose:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(f"{prefix}loading reset bank from cache: {resolved_cache_path}", flush=True)
            return self._load_bank(resolved_cache_path)

        compatible_cache_path = None
        if resolved_cache_path is not None and not force_rebuild:
            compatible_cache_path = self._find_compatible_cache_path(
                cache_path,
                bank_size_per_type,
            )
        if compatible_cache_path is not None:
            if verbose:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(
                    f"{prefix}loading compatible reset bank from cache: "
                    f"{compatible_cache_path}",
                    flush=True,
                )
            return self._load_bank(compatible_cache_path)

        if verbose:
            prefix = f"{log_prefix} " if log_prefix else ""
            if resolved_cache_path is not None:
                print(f"{prefix}building reset bank cache: {resolved_cache_path}", flush=True)
            else:
                print(f"{prefix}building reset bank in memory", flush=True)

        bank: Dict[str, Dict[str, np.ndarray]] = {}
        for reset_type in self.reset_types:
            samples: List[ResetState] = []
            attempts = 0
            max_attempts = max(bank_size_per_type * self.max_sample_attempts, self.max_sample_attempts)
            progress_stride = max(bank_size_per_type // 8, 1)
            if verbose:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(
                    f"{prefix}[{reset_type}] target={bank_size_per_type} max_attempts={max_attempts}",
                    flush=True,
                )
            while len(samples) < bank_size_per_type and attempts < max_attempts:
                attempts += 1
                sample = self._make_candidate(reset_type)
                if sample is not None:
                    samples.append(sample)
                    if verbose and (
                        len(samples) == 1
                        or len(samples) == bank_size_per_type
                        or len(samples) % progress_stride == 0
                    ):
                        prefix = f"{log_prefix} " if log_prefix else ""
                        print(
                            f"{prefix}[{reset_type}] {len(samples)}/{bank_size_per_type} "
                            f"(attempts={attempts})",
                            flush=True,
                        )
            if not samples:
                raise RuntimeError(f"failed to build OmniReset bank for '{reset_type}'")
            if len(samples) < bank_size_per_type:
                if verbose:
                    prefix = f"{log_prefix} " if log_prefix else ""
                    print(
                        f"{prefix}[{reset_type}] only found {len(samples)} samples; "
                        "duplicating to fill the bank",
                        flush=True,
                    )
                indices = self.rng.integers(0, len(samples), size=bank_size_per_type - len(samples))
                samples.extend([samples[int(index)] for index in indices])
            if verbose:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(
                    f"{prefix}[{reset_type}] done with {len(samples)} samples "
                    f"(attempts={attempts})",
                    flush=True,
                )
            bank[reset_type] = {
                "qpos": np.stack([sample.qpos for sample in samples], axis=0).astype(np.float32),
                "qvel": np.stack([sample.qvel for sample in samples], axis=0).astype(np.float32),
                "ctrl": np.stack([sample.ctrl for sample in samples], axis=0).astype(np.float32),
                "arm_qpos": np.stack(
                    [sample.qpos[self.arm_qpos_ids] for sample in samples], axis=0
                ).astype(np.float32),
                "gripper_qpos": np.asarray(
                    [sample.qpos[self.gripper_qpos_id] for sample in samples],
                    dtype=np.float32,
                )[:, None],
                "gripper_ctrl": np.asarray(
                    [sample.ctrl[6] if sample.ctrl.shape[0] > 6 else 0.0 for sample in samples],
                    dtype=np.float32,
                )[:, None],
            }
        if resolved_cache_path is not None:
            self._save_bank(resolved_cache_path, bank, bank_size_per_type)
            if verbose:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(f"{prefix}saved reset bank cache: {resolved_cache_path}", flush=True)
        return bank

    def sample_uniform_from_bank(self, bank: Dict[str, Dict[str, np.ndarray]]) -> ResetState:
        choices = np.asarray(self.reset_types)
        for _ in range(max(len(self.reset_types), 1)):
            reset_type = str(choices[int(self.rng.integers(0, len(choices)))])
            entry = bank.get(reset_type)
            if entry is None or int(entry["qpos"].shape[0]) <= 0:
                continue
            sample_index = int(self.rng.integers(0, entry["qpos"].shape[0]))
            return ResetState(
                reset_type=reset_type,
                qpos=np.asarray(entry["qpos"][sample_index], dtype=np.float64).copy(),
                qvel=np.asarray(entry["qvel"][sample_index], dtype=np.float64).copy(),
                ctrl=np.asarray(entry["ctrl"][sample_index], dtype=np.float64).copy(),
                sample_index=sample_index,
            )
        raise RuntimeError("reset bank does not contain any active reset type")

    def sample_batch(
        self,
        batch_size: int,
        bank: Dict[str, Dict[str, np.ndarray]] | None = None,
    ) -> Dict[str, np.ndarray | list[str]]:
        active_bank = bank
        if active_bank is None:
            active_bank = self.build_bank(1)

        qpos = np.empty((batch_size, self.model.nq), dtype=np.float32)
        qvel = np.empty((batch_size, self.model.nv), dtype=np.float32)
        ctrl = np.empty((batch_size, self.model.nu), dtype=np.float32)
        reset_types: list[str] = []

        choices = np.asarray(self.reset_types)
        for batch_index in range(batch_size):
            reset_type = str(choices[int(self.rng.integers(0, len(choices)))])
            entry = active_bank[reset_type]
            sample_index = int(self.rng.integers(0, entry["qpos"].shape[0]))
            qpos[batch_index] = entry["qpos"][sample_index]
            qvel[batch_index] = entry["qvel"][sample_index]
            ctrl[batch_index] = entry["ctrl"][sample_index]
            reset_types.append(reset_type)

        return {"qpos": qpos, "qvel": qvel, "ctrl": ctrl, "reset_types": reset_types}
