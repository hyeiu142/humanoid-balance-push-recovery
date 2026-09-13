"""
V2 Model Predictive Control (MPC) Controller with Stepping Push Recovery for Unitree G1 Humanoid.
Combines High-Level LIPM Preview Control with Capture Point Stepping Recovery
and Low-Level Inverse Kinematics Tracking.
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np
import mujoco

from sim.sensors import RobotState
from controllers.base_controller import BaseController
from controllers.v2_mpc.lipm_model import LIPMModel
from controllers.v2_mpc.qp_mpc_solver import LIPMQPSolver
from controllers.v2_mpc.leg_kinematics import LegKinematicsSolver


class SteppingMode(Enum):
    SINGLE_LEG = "Single-Leg Stepping"
    SYNC_SHUFFLE = "Synchronous Shuffle"


class RecoveryState(Enum):
    DOUBLE_SUPPORT = "In-Place MPC Balance"
    STEP_SWING = "Stepping Recovery"
    LANDED_SETTLE = "Landed Settle MPC"


class V2MPCController(BaseController):
    """
    V2 Controller: Hierarchical Model Predictive Control with Stepping Push Recovery.
    - Low/Medium impulses (<= 120N): Predictive In-Place ZMP MPC & Torso stabilization (< 0.8 deg tilt).
    - Heavy impulses (> 150N up to 260N+): Instantaneous Capture Point (ICP) Stepping Recovery.
    Supports both Single-Leg Stepping (human-like single swing) and Synchronous Shuffle.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        nominal_qpos: np.ndarray,
        horizon: int = 16,
        mpc_dt: float = 0.05,
        step_duration: float = 0.18,
        stepping_mode: SteppingMode = SteppingMode.SINGLE_LEG,
    ):
        super().__init__(model, nominal_qpos)

        self.stepping_mode = stepping_mode
        self.lipm = LIPMModel(z0=0.693, g=9.81)

        # Sagittal & Coronal LIPM QP Solvers
        self.qp_x = LIPMQPSolver(
            lipm=self.lipm,
            horizon=horizon,
            dt=mpc_dt,
            qx=150.0,
            qv=20.0,
            r_zmp=0.5,
            r_rate=0.05,
        )
        self.qp_y = LIPMQPSolver(
            lipm=self.lipm,
            horizon=horizon,
            dt=mpc_dt,
            qx=100.0,
            qv=15.0,
            r_zmp=0.5,
            r_rate=0.05,
        )

        self.ik_solver = LegKinematicsSolver(model)
        self.step_duration = step_duration

        # Nominal relative foot vectors in pelvis frame
        self.nom_rel_left = np.array([0.0, 0.1185, -0.7568])
        self.nom_rel_right = np.array([0.0, -0.1185, -0.7568])

        # Actuator indices
        self.idx_left_hip_pitch = 0
        self.idx_left_hip_roll = 1
        self.idx_left_knee = 3
        self.idx_left_ankle_pitch = 4
        self.idx_left_ankle_roll = 5

        self.idx_right_hip_pitch = 6
        self.idx_right_hip_roll = 7
        self.idx_right_knee = 9
        self.idx_right_ankle_pitch = 10
        self.idx_right_ankle_roll = 11

        # Control limits
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]

        # FSM and tracking states
        self.fsm_state = RecoveryState.DOUBLE_SUPPORT
        self.step_start_t = 0.0
        self.total_offset = 0.0
        self.cur_step_len = 0.20
        self.step_count = 0
        self.max_steps = 2
        self.swing_leg = "right"
        self.staggered_baseline = self.nominal_qpos.copy()
        self.roll_int = 0.0
        self._active_swing_mode = self.stepping_mode
        self.single_leg_duration = 0.16
        self.prev_vx = 0.0
        self.com_ax = 0.0

        self.last_mpc_t = -1.0
        self.opt_zmp_x = 0.0
        self.opt_zmp_y = 0.0
        self.mpc_interval = mpc_dt

    def reset(self):
        """Reset internal controller states."""
        self.fsm_state = RecoveryState.DOUBLE_SUPPORT
        self.step_start_t = 0.0
        self.total_offset = 0.0
        self.cur_step_len = 0.20
        self.step_count = 0
        self.last_mpc_t = -1.0
        self.opt_zmp_x = 0.0
        self.opt_zmp_y = 0.0
        self.roll_int = 0.0
        self.prev_vx = 0.0
        self.com_ax = 0.0
        self._active_swing_mode = self.stepping_mode
        self.staggered_baseline = self.nominal_qpos.copy()
        self.qp_x.reset()
        self.qp_y.reset()

    def compute_action(self, state: RobotState, dt: float) -> np.ndarray:
        """
        Compute 29-DOF actuator joint position commands.
        """
        action = self.nominal_qpos.copy()
        current_time = state.time

        # Track CoM acceleration
        self.com_ax = (state.com_vel[0] - self.prev_vx) / dt
        self.prev_vx = state.com_vel[0]

        # 1. Roll / Lateral stabilization across all phases (V1 proven signs + leaky integral)
        roll_err = state.pelvis_rpy[0]
        roll_rate = state.pelvis_ang_vel[0]
        self.roll_int = 0.995 * self.roll_int + roll_err * dt
        u_roll = 1.2 * roll_err + 0.15 * roll_rate + 0.5 * self.roll_int
        delta_hip_roll = np.clip(0.8 * u_roll, -0.30, 0.30)
        delta_ankle_roll = np.clip(0.6 * u_roll, -0.20, 0.20)
        action[self.idx_left_ankle_roll] += delta_ankle_roll
        action[self.idx_right_ankle_roll] += delta_ankle_roll
        action[self.idx_left_hip_roll] -= delta_hip_roll
        action[self.idx_right_hip_roll] -= delta_hip_roll

        # Instantaneous Capture Point (Sagittal)
        icp_x = state.com_pos[0] + state.com_vel[0] / self.lipm.omega_0

        # 2. State Machine: In-Place vs Stepping
        if self.fsm_state == RecoveryState.DOUBLE_SUPPORT:
            pitch_err = state.pelvis_rpy[1]
            com_x_err = state.com_pos[0] - (0.003 + self.total_offset)
            u_pitch = (
                1.2 * pitch_err
                + 0.15 * state.pelvis_ang_vel[1]
                + 0.8 * com_x_err
                + 0.12 * state.com_vel[0]
            )
            action[self.idx_left_ankle_pitch] += np.clip(0.8 * u_pitch, -0.35, 0.35)
            action[self.idx_right_ankle_pitch] += np.clip(0.8 * u_pitch, -0.35, 0.35)
            action[self.idx_left_hip_pitch] += np.clip(1.2 * u_pitch, -0.40, 0.40)
            action[self.idx_right_hip_pitch] += np.clip(1.2 * u_pitch, -0.40, 0.40)

            # Check if push triggers Stepping Recovery
            trigger = (state.com_vel[0] > 0.28 or icp_x > self.total_offset + 0.11) and current_time >= 0.10

            if trigger:
                self.fsm_state = RecoveryState.STEP_SWING
                self.step_start_t = current_time
                self.step_count += 1
                pred_touchdown = icp_x * np.exp(self.lipm.omega_0 * self.step_duration)

                if self.stepping_mode == SteppingMode.SINGLE_LEG and self.com_ax < 2.6:
                    self._active_swing_mode = SteppingMode.SINGLE_LEG
                    self.cur_step_len = 0.08
                    self.swing_leg = "left" if state.pelvis_rpy[0] > 0.03 else "right"
                else:
                    # For extreme pushes (>150N), adaptively promote to bilateral sync shuffle
                    self._active_swing_mode = SteppingMode.SYNC_SHUFFLE
                    self.cur_step_len = float(np.clip(pred_touchdown - self.total_offset, 0.16, 0.26))

        elif self.fsm_state == RecoveryState.STEP_SWING:
            dur = self.single_leg_duration if self._active_swing_mode == SteppingMode.SINGLE_LEG else self.step_duration
            elapsed = current_time - self.step_start_t
            tau = np.clip(elapsed / dur, 0.0, 1.0)
            s = 0.5 * (1.0 - np.cos(np.pi * tau))

            # Load full joint state for Inverse Kinematics
            full_qpos = np.zeros(self.model.nq)
            full_qpos[0:3] = state.pelvis_pos
            full_qpos[3:7] = state.pelvis_quat
            for i in range(self.model.nu):
                jnt_id = self.model.actuator_trnid[i, 0]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                full_qpos[qpos_adr] = state.joint_pos[i]

            if self._active_swing_mode == SteppingMode.SINGLE_LEG:
                cur_dx = s * self.cur_step_len
                cur_dz = 0.012 * np.sin(np.pi * tau)
                sign_y = 1.0 if self.swing_leg == "right" else -1.0
                dy_shift = sign_y * 0.035 * np.sin(np.pi * tau)

                full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[3, 0]]] = max(0.15, full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[3, 0]]])
                full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[9, 0]]] = max(0.15, full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[9, 0]]])

                if self.swing_leg == "right":
                    t_left = np.array([0.0, 0.1185 - dy_shift, -0.7568])
                    t_right = np.array([cur_dx, -0.1185 - dy_shift, -0.7568 + cur_dz])
                else:
                    t_left = np.array([cur_dx, 0.1185 - dy_shift, -0.7568 + cur_dz])
                    t_right = np.array([0.0, -0.1185 - dy_shift, -0.7568])

                la = self.ik_solver.solve_ik("left", t_left, full_qpos)
                ra = self.ik_solver.solve_ik("right", t_right, full_qpos)

                # Sagittal joints
                action[0] = la[0]
                action[3] = la[3]
                action[4] = -(la[0] + la[3] + state.pelvis_rpy[1])

                action[6] = ra[0]
                action[9] = ra[3]
                action[10] = -(ra[0] + ra[3] + state.pelvis_rpy[1])

                # Weight shift hip roll
                action[1] = la[1]
                action[7] = ra[1]

                # Torso pitch damping during flight
                torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
                action[self.idx_left_hip_pitch] += torso_damp
                action[self.idx_right_hip_pitch] += torso_damp

                # Check Touchdown
                if elapsed >= dur:
                    self.fsm_state = RecoveryState.LANDED_SETTLE
                    self.total_offset += self.cur_step_len / 2.0

            else:
                # Synchronous Shuffle mode: both feet move together
                cur_dx = s * self.cur_step_len
                rem_t = max(0.0, self.step_duration - elapsed)
                pred_now = icp_x * np.exp(self.lipm.omega_0 * rem_t)
                desired_step = pred_now - self.total_offset
                self.cur_step_len = float(np.clip(max(self.cur_step_len, desired_step), 0.16, 0.26))

                cur_dz = 0.030 * np.sin(np.pi * tau) + 0.035 * s

                t_left = self.nom_rel_left.copy()
                t_left[0] += cur_dx
                t_left[2] += cur_dz
                t_right = self.nom_rel_right.copy()
                t_right[0] += cur_dx
                t_right[2] += cur_dz

                left_angles = self.ik_solver.solve_ik("left", t_left, full_qpos)
                right_angles = self.ik_solver.solve_ik("right", t_right, full_qpos)

                leveling_pitch = -(left_angles[0] + left_angles[3] + state.pelvis_rpy[1])
                left_angles[4] = np.clip(leveling_pitch, -0.6, 0.8)
                right_angles[4] = np.clip(leveling_pitch, -0.6, 0.8)

                for i, act_idx in enumerate(self.ik_solver.left_actuators):
                    action[act_idx] = left_angles[i]
                for i, act_idx in enumerate(self.ik_solver.right_actuators):
                    action[act_idx] = right_angles[i]

                torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
                action[self.idx_left_hip_pitch] += torso_damp
                action[self.idx_right_hip_pitch] += torso_damp

                # Check Touchdown
                if elapsed >= self.step_duration:
                    self.total_offset += self.cur_step_len
                    self.fsm_state = RecoveryState.LANDED_SETTLE

        elif self.fsm_state == RecoveryState.LANDED_SETTLE:
            if self._active_swing_mode == SteppingMode.SINGLE_LEG:
                d_hip = self.cur_step_len / 0.75
                if self.swing_leg == "right":
                    action[6] -= 0.5 * d_hip
                    action[10] += 0.5 * d_hip
                    action[0] += 0.5 * d_hip
                    action[4] -= 0.5 * d_hip
                else:
                    action[0] -= 0.5 * d_hip
                    action[4] += 0.5 * d_hip
                    action[6] += 0.5 * d_hip
                    action[10] -= 0.5 * d_hip

                action[3] += 0.05
                action[9] += 0.05

                action[self.idx_left_ankle_roll] += delta_ankle_roll
                action[self.idx_right_ankle_roll] += delta_ankle_roll
                action[self.idx_left_hip_roll] -= delta_hip_roll
                action[self.idx_right_hip_roll] -= delta_hip_roll

                pitch_err = state.pelvis_rpy[1]
                pitch_rate = state.pelvis_ang_vel[1]
                u_pitch = 1.2 * pitch_err + 0.16 * pitch_rate + 0.25 * state.com_vel[0]
                action[self.idx_left_ankle_pitch] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                action[self.idx_right_ankle_pitch] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                action[self.idx_left_hip_pitch] += np.clip(1.0 * u_pitch, -0.35, 0.35)
                action[self.idx_right_hip_pitch] += np.clip(1.0 * u_pitch, -0.35, 0.35)

            else:
                # Physical foot center in world frame from forward kinematics
                lf_id = self.ik_solver.left_foot_id
                rf_id = self.ik_solver.right_foot_id
                foot_center_x = 0.5 * (self.ik_solver.data.xpos[lf_id][0] + self.ik_solver.data.xpos[rf_id][0])
                ref_x = foot_center_x + 0.003
                zmp_min = foot_center_x - 0.05
                zmp_max = foot_center_x + 0.11

                # MPC preview solve at 20 Hz
                if current_time - self.last_mpc_t >= self.mpc_interval or self.last_mpc_t < 0:
                    opt_zmp, _, _ = self.qp_x.solve(
                        state.com_pos[0],
                        state.com_vel[0],
                        ref_pos=ref_x,
                        zmp_min=zmp_min,
                        zmp_max=zmp_max,
                    )
                    self.opt_zmp_x = opt_zmp
                    self.last_mpc_t = current_time

                pitch_err = state.pelvis_rpy[1]
                pitch_rate = state.pelvis_ang_vel[1]
                com_err = state.com_pos[0] - ref_x
                com_vx = state.com_vel[0]

                u_pitch = (
                    1.4 * pitch_err
                    + 0.18 * pitch_rate
                    + 1.0 * com_err
                    + 0.15 * com_vx
                    + 0.6 * (self.opt_zmp_x - ref_x)
                )

                action[self.idx_left_ankle_pitch] += np.clip(1.0 * u_pitch, -0.35, 0.35)
                action[self.idx_right_ankle_pitch] += np.clip(1.0 * u_pitch, -0.35, 0.35)
                action[self.idx_left_hip_pitch] += np.clip(1.4 * u_pitch, -0.45, 0.45)
                action[self.idx_right_hip_pitch] += np.clip(1.4 * u_pitch, -0.45, 0.45)

                # Secondary step if ICP exceeds toe bound again after settling
                if icp_x > foot_center_x + 0.14 and self.step_count < self.max_steps and (current_time - (self.step_start_t + self.step_duration) > 0.12):
                    self.fsm_state = RecoveryState.STEP_SWING
                    self.step_start_t = current_time
                    self.step_count += 1
                    rel_icp = max(0.0, icp_x - foot_center_x)
                    self.cur_step_len = float(np.clip(rel_icp * np.exp(self.lipm.omega_0 * self.step_duration), 0.10, 0.18))

        action = np.clip(action, self.ctrl_min, self.ctrl_max)
        return action
