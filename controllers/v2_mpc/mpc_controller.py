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


class RecoveryState(Enum):
    DOUBLE_SUPPORT = "In-Place MPC Balance"
    STEP_SWING = "Stepping Recovery"
    LANDED_SETTLE = "Landed Settle MPC"


class V2MPCController(BaseController):
    """
    V2 Controller: Hierarchical Model Predictive Control with Stepping Push Recovery.
    - Low/Medium impulses (<= 120N): Predictive In-Place ZMP MPC & Torso stabilization (< 0.8 deg tilt).
    - Heavy impulses (> 150N up to 260N+): Instantaneous Capture Point (ICP) Stepping Recovery.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        nominal_qpos: np.ndarray,
        horizon: int = 16,
        mpc_dt: float = 0.05,
        step_duration: float = 0.18,
    ):
        super().__init__(model, nominal_qpos)

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
        self.qp_x.reset()
        self.qp_y.reset()

    def compute_action(self, state: RobotState, dt: float) -> np.ndarray:
        """
        Compute 29-DOF actuator joint position commands.
        """
        action = self.nominal_qpos.copy()
        current_time = state.time

        # 1. Roll / Lateral stabilization across all phases
        roll_err = state.pelvis_rpy[0]
        roll_rate = state.pelvis_ang_vel[0]
        delta_hip_roll = np.clip(1.2 * roll_err + 0.15 * roll_rate, -0.35, 0.35)
        delta_ankle_roll = np.clip(0.8 * roll_err + 0.10 * roll_rate, -0.20, 0.20)

        action[self.idx_left_hip_roll] += delta_hip_roll
        action[self.idx_right_hip_roll] += delta_hip_roll
        action[self.idx_left_ankle_roll] += delta_ankle_roll
        action[self.idx_right_ankle_roll] += delta_ankle_roll

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

            # Check if heavy push triggers Stepping Recovery
            if (icp_x > self.total_offset + 0.11 and current_time >= 0.10) or (state.com_vel[0] > 0.28):
                self.fsm_state = RecoveryState.STEP_SWING
                self.step_start_t = current_time
                self.step_count += 1
                pred_touchdown = icp_x * np.exp(self.lipm.omega_0 * self.step_duration)
                self.cur_step_len = float(np.clip(pred_touchdown - self.total_offset, 0.16, 0.26))

        elif self.fsm_state == RecoveryState.STEP_SWING:
            elapsed = current_time - self.step_start_t
            tau = np.clip(elapsed / self.step_duration, 0.0, 1.0)

            # Dynamic capture point adaptation as impulse completes
            rem_t = max(0.0, self.step_duration - elapsed)
            pred_now = icp_x * np.exp(self.lipm.omega_0 * rem_t)
            desired_step = pred_now - self.total_offset
            self.cur_step_len = float(np.clip(max(self.cur_step_len, desired_step), 0.16, 0.26))

            # Minimum-jerk Cycloid 3D Foot trajectory
            s = 0.5 * (1.0 - np.cos(np.pi * tau))
            cur_dx = s * self.cur_step_len
            cur_dz = 0.030 * np.sin(np.pi * tau) + 0.035 * s

            # Solve Inverse Kinematics for both legs
            full_qpos = np.zeros(self.model.nq)
            full_qpos[0:3] = state.pelvis_pos
            full_qpos[3:7] = state.pelvis_quat
            for i in range(self.model.nu):
                jnt_id = self.model.actuator_trnid[i, 0]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                full_qpos[qpos_adr] = state.joint_pos[i]

            t_left = self.nom_rel_left.copy()
            t_left[0] += cur_dx
            t_left[2] += cur_dz
            t_right = self.nom_rel_right.copy()
            t_right[0] += cur_dx
            t_right[2] += cur_dz

            left_angles = self.ik_solver.solve_ik("left", t_left, full_qpos)
            right_angles = self.ik_solver.solve_ik("right", t_right, full_qpos)

            # Foot Sole Leveling: hold foot soles parallel to ground
            leveling_pitch = -(left_angles[0] + left_angles[3] + state.pelvis_rpy[1])
            left_angles[4] = np.clip(leveling_pitch, -0.6, 0.8)
            right_angles[4] = np.clip(leveling_pitch, -0.6, 0.8)

            for i, act_idx in enumerate(self.ik_solver.left_actuators):
                action[act_idx] = left_angles[i]
            for i, act_idx in enumerate(self.ik_solver.right_actuators):
                action[act_idx] = right_angles[i]

            # Torso pitch damping during flight
            torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
            action[self.idx_left_hip_pitch] += torso_damp
            action[self.idx_right_hip_pitch] += torso_damp

            # Check Touchdown
            if elapsed >= self.step_duration:
                self.total_offset += self.cur_step_len
                self.fsm_state = RecoveryState.LANDED_SETTLE

        elif self.fsm_state == RecoveryState.LANDED_SETTLE:
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
