"""
Virtual Model Control (VMC) / Task-Space PID Controller for Humanoid Balance.
Controls torso orientation and CoM via coordinated Ankle and Hip strategies.
"""

from typing import Dict, Any, Tuple
import numpy as np
import mujoco

from sim.sensors import RobotState
from controllers.base_controller import BaseController
from controllers.v1_pid.balance_strategies import StrategyCoordinator, BalanceStrategyMode


class VMCPIDController(BaseController):
    """
    V1 Controller: Virtual Model Control with Ankle and Hip strategies.
    Computes restorative joint adjustments based on virtual springs and dampers on the torso & CoM.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        nominal_qpos: np.ndarray,
        kp_pitch: float = 1.2,
        kd_pitch: float = 0.15,
        ki_pitch: float = 0.05,
        kp_roll: float = 0.9,
        kd_roll: float = 0.12,
        ki_roll: float = 0.03,
        kp_com_x: float = 0.8,
        kd_com_x: float = 0.1,
        kp_com_y: float = 0.6,
        kd_com_y: float = 0.08,
        kp_height: float = 1.0,
        kd_height: float = 0.15,
    ):
        super().__init__(model, nominal_qpos)

        # PID Gains for Sagittal (Pitch / X)
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.ki_pitch = ki_pitch
        self.kp_com_x = kp_com_x
        self.kd_com_x = kd_com_x

        # PID Gains for Coronal (Roll / Y)
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.ki_roll = ki_roll
        self.kp_com_y = kp_com_y
        self.kd_com_y = kd_com_y

        # Gains for CoM Height (Z)
        self.kp_height = kp_height
        self.kd_height = kd_height

        # Integrator states
        self.pitch_integral = 0.0
        self.roll_integral = 0.0
        self.integral_limit = 0.15  # anti-windup (rad)

        # Desired setpoints (nominal standing)
        self.des_pitch = 0.0
        self.des_roll = 0.0
        self.des_com_x = 0.003
        self.des_com_y = 0.0
        self.des_com_z = 0.693  # default CoM height

        # Actuator indices in Unitree G1:
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

        # Actuator control limits from model
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]

        # Strategy coordinator
        self.coordinator = StrategyCoordinator()

    def reset(self):
        """Reset PID integrators and strategy coordinator."""
        self.pitch_integral = 0.0
        self.roll_integral = 0.0
        self.coordinator = StrategyCoordinator()

    def compute_action(self, state: RobotState, dt: float) -> np.ndarray:
        """
        Compute target joint positions for all 29 actuators.
        """
        # Start from nominal standing posture
        action = self.nominal_qpos.copy()

        # 1. State errors
        pitch_err = state.pelvis_rpy[1] - self.des_pitch
        pitch_rate = state.pelvis_ang_vel[1]
        com_x_err = state.com_pos[0] - self.des_com_x
        com_vx = state.com_vel[0]

        roll_err = state.pelvis_rpy[0] - self.des_roll
        roll_rate = state.pelvis_ang_vel[0]
        com_y_err = state.com_pos[1] - self.des_com_y
        com_vy = state.com_vel[1]

        height_err = state.com_pos[2] - self.des_com_z
        height_vz = state.com_vel[2]

        # Update integrals with anti-windup
        self.pitch_integral = np.clip(
            self.pitch_integral + pitch_err * dt,
            -self.integral_limit,
            self.integral_limit
        )
        self.roll_integral = np.clip(
            self.roll_integral + roll_err * dt,
            -self.integral_limit,
            self.integral_limit
        )

        # 2. Strategy evaluation
        total_tilt_err = np.sqrt(pitch_err**2 + roll_err**2)
        total_tilt_vel = np.sqrt(pitch_rate**2 + roll_rate**2)
        mode = self.coordinator.evaluate(
            tilt_error=total_tilt_err,
            tilt_vel=total_tilt_vel,
            cop_margin=state.cop_margin
        )
        w_ankle = self.coordinator.ankle_weight
        w_hip = self.coordinator.hip_weight

        # 3. Virtual Restoring Torques / Signals
        # Pitch (Sagittal): positive pitch_err means leaning forward -> need positive ankle pitch (pushes toes down)
        u_pitch = (
            self.kp_pitch * pitch_err
            + self.kd_pitch * pitch_rate
            + self.kp_com_x * com_x_err
            + self.kd_com_x * com_vx
            + self.ki_pitch * self.pitch_integral
        )

        # Roll (Coronal): positive roll_err means leaning right -> roll ankles to shift CoP
        u_roll = (
            self.kp_roll * roll_err
            + self.kd_roll * roll_rate
            + self.kp_com_y * com_y_err
            + self.kd_com_y * com_vy
            + self.ki_roll * self.roll_integral
        )

        # Height (Z): error in CoM height -> knee adjustment
        u_height = (
            self.kp_height * height_err
            + self.kd_height * height_vz
        )

        # 4. Joint adjustments based on Ankle & Hip Strategy:

        # --- Ankle Strategy ---
        # Ankle pitch rotates foot to shift CoP along X
        delta_ankle_pitch = w_ankle * np.clip(0.8 * u_pitch, -0.35, 0.35)
        # Ankle roll rotates foot to shift CoP along Y
        delta_ankle_roll = w_ankle * np.clip(0.6 * u_roll, -0.20, 0.20)

        action[self.idx_left_ankle_pitch] += delta_ankle_pitch
        action[self.idx_right_ankle_pitch] += delta_ankle_pitch

        action[self.idx_left_ankle_roll] += delta_ankle_roll
        action[self.idx_right_ankle_roll] += delta_ankle_roll

        # --- Hip Strategy ---
        # Hip pitch bends torso to generate angular momentum counter-reaction
        # When pushed forward, torso flexes forward (- or + hip pitch depending on axis)
        delta_hip_pitch = w_hip * np.clip(1.2 * u_pitch, -0.5, 0.5)
        action[self.idx_left_hip_pitch] += delta_hip_pitch
        action[self.idx_right_hip_pitch] += delta_hip_pitch

        # Hip roll tilts pelvis laterally
        delta_hip_roll = w_hip * np.clip(0.8 * u_roll, -0.3, 0.3)
        action[self.idx_left_hip_roll] -= delta_hip_roll
        action[self.idx_right_hip_roll] -= delta_hip_roll

        # --- Stance Height / Knee PD ---
        # If height drops or in large hip flexion, adjust knee to absorb shock / maintain height
        delta_knee = np.clip(0.5 * u_height, -0.2, 0.2)
        action[self.idx_left_knee] += delta_knee
        action[self.idx_right_knee] += delta_knee

        # 5. Enforce joint limits
        action = np.clip(action, self.ctrl_min, self.ctrl_max)
        return action
