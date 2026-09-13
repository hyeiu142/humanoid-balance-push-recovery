"""
V2 Model Predictive Control (MPC) Controller with Stepping Push Recovery for Unitree G1 Humanoid.
Combines High-Level LIPM Preview Control with Capture Point Stepping Recovery
and Low-Level Inverse Kinematics Tracking.
"""

from enum import Enum
import numpy as np
import mujoco

from sim.sensors import RobotState
from controllers.base_controller import BaseController
from controllers.v2_mpc.lipm_model import LIPMModel
from controllers.v2_mpc.qp_mpc_solver import LIPMQPSolver
from controllers.v2_mpc.leg_kinematics import LegKinematicsSolver
from controllers.v2_mpc.footstep_planner import FootstepPlanner, StepPlan, SteppingMode
from controllers.v2_mpc.swing_trajectory import SwingTrajectoryGenerator


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
        self.footstep_planner = FootstepPlanner(
            lipm=self.lipm,
            step_duration=step_duration,
            single_leg_duration=0.16,
        )
        self.single_leg_trajectory = SwingTrajectoryGenerator(step_height=0.022)
        self.shuffle_trajectory = SwingTrajectoryGenerator(step_height=0.030)
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
        self.settle_start_t = 0.0
        self.total_offset = 0.0
        self.cur_step_len = 0.20
        self.step_count = 0
        self.max_steps = 2
        self.swing_leg = "right"
        self.staggered_baseline = self.nominal_qpos.copy()
        self.roll_int = 0.0
        self._active_swing_mode = self.stepping_mode
        self.single_leg_duration = 0.16
        self.active_step_duration = step_duration
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
        self.settle_start_t = 0.0
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
        self.active_step_duration = self.step_duration
        self.staggered_baseline = self.nominal_qpos.copy()
        self.footstep_planner.reset()
        self.qp_x.reset()
        self.qp_y.reset()

    def _start_step(self, plan: StepPlan, current_time: float) -> None:
        """Commit one planner decision to the execution FSM."""
        self.fsm_state = RecoveryState.STEP_SWING
        self.step_start_t = current_time
        self.step_count += 1
        self._active_swing_mode = plan.mode
        self.active_step_duration = plan.duration
        self.cur_step_len = plan.step_length
        self.swing_leg = plan.swing_leg

    def _landing_confirmed(self, state: RobotState, elapsed: float, duration: float) -> bool:
        """Use contact when available, with a bounded timeout fallback."""
        if elapsed < duration:
            return False

        if self._active_swing_mode == SteppingMode.SINGLE_LEG:
            contact = (
                state.left_foot_contact
                if self.swing_leg == "left"
                else state.right_foot_contact
            )
        else:
            contact = state.left_foot_contact and state.right_foot_contact

        return contact or elapsed >= duration + 0.05

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

            # The planner owns trigger, mode, leg selection, and step length.
            plan = self.footstep_planner.plan(
                current_time=current_time,
                com_pos_x=state.com_pos[0],
                com_vel_x=state.com_vel[0],
                com_acc_x=self.com_ax,
                pelvis_roll=state.pelvis_rpy[0],
                total_offset=self.total_offset,
                stepping_mode=self.stepping_mode,
            )
            if plan is not None:
                self._start_step(plan, current_time)

        elif self.fsm_state == RecoveryState.STEP_SWING:
            dur = self.active_step_duration
            elapsed = current_time - self.step_start_t
            # Load full joint state for Inverse Kinematics
            full_qpos = np.zeros(self.model.nq)
            full_qpos[0:3] = state.pelvis_pos
            full_qpos[3:7] = state.pelvis_quat
            for i in range(self.model.nu):
                jnt_id = self.model.actuator_trnid[i, 0]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                full_qpos[qpos_adr] = state.joint_pos[i]

            if self._active_swing_mode == SteppingMode.SINGLE_LEG:
                sign_y = 1.0 if self.swing_leg == "right" else -1.0
                midpoint_offset = np.array([0.0, -sign_y * 0.038, 0.0])

                full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[3, 0]]] = max(
                    0.15,
                    full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[3, 0]]],
                )
                full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[9, 0]]] = max(
                    0.15,
                    full_qpos[self.model.jnt_qposadr[self.model.actuator_trnid[9, 0]]],
                )

                left_start = self.nom_rel_left.copy()
                right_start = self.nom_rel_right.copy()
                left_target = left_start.copy()
                right_target = right_start.copy()
                if self.swing_leg == "right":
                    right_target[0] += self.cur_step_len
                else:
                    left_target[0] += self.cur_step_len

                t_left, _ = self.single_leg_trajectory.evaluate(
                    left_start,
                    left_target,
                    elapsed,
                    dur,
                    midpoint_offset=midpoint_offset,
                )
                t_right, _ = self.single_leg_trajectory.evaluate(
                    right_start,
                    right_target,
                    elapsed,
                    dur,
                    midpoint_offset=midpoint_offset,
                )

                la = self.ik_solver.solve_ik("left", t_left, full_qpos)
                ra = self.ik_solver.solve_ik("right", t_right, full_qpos)

                # Merge the IK result with the explicit stabilization ownership.
                action[self.idx_left_hip_pitch] = la[0]
                action[self.idx_left_knee] = la[3]
                action[self.idx_left_ankle_pitch] = -(
                    la[0] + la[3] + state.pelvis_rpy[1]
                )

                action[self.idx_right_hip_pitch] = ra[0]
                action[self.idx_right_knee] = ra[3]
                action[self.idx_right_ankle_pitch] = -(
                    ra[0] + ra[3] + state.pelvis_rpy[1]
                )

                # Weight shift hip roll.
                action[self.idx_left_hip_roll] = la[1]
                action[self.idx_right_hip_roll] = ra[1]

                # Torso pitch damping during flight
                torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
                action[self.idx_left_hip_pitch] += torso_damp
                action[self.idx_right_hip_pitch] += torso_damp

                # Check Touchdown
                if self._landing_confirmed(state, elapsed, dur):
                    self.fsm_state = RecoveryState.LANDED_SETTLE
                    self.settle_start_t = current_time
                    self.total_offset += self.cur_step_len / 2.0
                    self.roll_int = 0.0

            else:
                # Synchronous Shuffle mode: both feet move together
                rem_t = max(0.0, self.active_step_duration - elapsed)
                pred_now = icp_x * np.exp(self.lipm.omega_0 * rem_t)
                desired_step = pred_now - self.total_offset
                self.cur_step_len = float(np.clip(max(self.cur_step_len, desired_step), 0.16, 0.26))

                left_start = self.nom_rel_left.copy()
                right_start = self.nom_rel_right.copy()
                left_target = left_start.copy()
                right_target = right_start.copy()
                left_target[0] += self.cur_step_len
                right_target[0] += self.cur_step_len
                left_target[2] += 0.035
                right_target[2] += 0.035

                t_left, _ = self.shuffle_trajectory.evaluate(
                    left_start,
                    left_target,
                    elapsed,
                    self.active_step_duration,
                )
                t_right, _ = self.shuffle_trajectory.evaluate(
                    right_start,
                    right_target,
                    elapsed,
                    self.active_step_duration,
                )

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
                if self._landing_confirmed(state, elapsed, dur):
                    self.total_offset += self.cur_step_len
                    self.fsm_state = RecoveryState.LANDED_SETTLE
                    self.settle_start_t = current_time

        elif self.fsm_state == RecoveryState.LANDED_SETTLE:
            if self._active_swing_mode == SteppingMode.SINGLE_LEG:
                d_hip = self.cur_step_len / 0.75
                if self.swing_leg == "right":
                    action[self.idx_right_hip_pitch] -= 0.5 * d_hip
                    action[self.idx_right_ankle_pitch] += 0.5 * d_hip
                    action[self.idx_left_hip_pitch] += 0.5 * d_hip
                    action[self.idx_left_ankle_pitch] -= 0.5 * d_hip
                else:
                    action[self.idx_left_hip_pitch] -= 0.5 * d_hip
                    action[self.idx_left_ankle_pitch] += 0.5 * d_hip
                    action[self.idx_right_hip_pitch] += 0.5 * d_hip
                    action[self.idx_right_ankle_pitch] -= 0.5 * d_hip

                action[self.idx_left_knee] += 0.05
                action[self.idx_right_knee] += 0.05

                pitch_err = state.pelvis_rpy[1]
                pitch_rate = state.pelvis_ang_vel[1]
                u_pitch = 1.2 * pitch_err + 0.16 * pitch_rate + 0.25 * state.com_vel[0]
                action[self.idx_left_ankle_pitch] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                action[self.idx_right_ankle_pitch] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                action[self.idx_left_hip_pitch] += np.clip(1.0 * u_pitch, -0.35, 0.35)
                action[self.idx_right_hip_pitch] += np.clip(1.0 * u_pitch, -0.35, 0.35)

            else:
                # Physical foot center in world frame from the live state.
                foot_center_x = 0.5 * (
                    state.left_foot_pos[0] + state.right_foot_pos[0]
                )
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

                # The planner also owns the secondary-step policy.
                plan = self.footstep_planner.plan_secondary_shuffle(
                    icp_x=icp_x,
                    foot_center_x=foot_center_x,
                    current_time=current_time,
                    last_step_start_time=self.step_start_t,
                    step_count=self.step_count,
                    max_steps=self.max_steps,
                )
                if plan is not None:
                    self._start_step(plan, current_time)

            if (
                self.fsm_state == RecoveryState.LANDED_SETTLE
                and current_time - self.settle_start_t >= 0.25
                and abs(state.pelvis_rpy[0]) < 0.08
                and abs(state.pelvis_rpy[1]) < 0.08
                and abs(state.com_vel[0]) < 0.10
            ):
                self.fsm_state = RecoveryState.DOUBLE_SUPPORT

        action = np.clip(action, self.ctrl_min, self.ctrl_max)
        return action
