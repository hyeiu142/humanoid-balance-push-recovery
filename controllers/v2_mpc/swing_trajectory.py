"""
Swing Foot Trajectory Generator for Stepping Push Recovery.
Generates smooth 3D minimum-jerk / cycloid trajectories for the swing foot
from liftoff to touchdown with clearance height at apex.
"""

import numpy as np


class SwingTrajectoryGenerator:
    """
    Generates 3D trajectory [pos, vel] for a swing foot during a single step.
    """

    def __init__(self, step_height: float = 0.06):
        self.step_height = float(step_height)  # Apex clearance height (6 cm)

    def evaluate(
        self,
        p_start: np.ndarray,
        p_target: np.ndarray,
        time_elapsed: float,
        step_duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate swing foot position and velocity at time_elapsed in [0, step_duration].
        Returns:
            pos: np.ndarray (3,) [x, y, z]
            vel: np.ndarray (3,) [vx, vy, vz]
        """
        if step_duration <= 0.0:
            return p_target.copy(), np.zeros(3)

        tau = np.clip(time_elapsed / step_duration, 0.0, 1.0)

        # Smooth S-curve interpolation for X, Y: s(tau) = 0.5 * (1 - cos(pi * tau))
        s = 0.5 * (1.0 - np.cos(np.pi * tau))
        s_dot = 0.5 * (np.pi / step_duration) * np.sin(np.pi * tau)

        pos_xy = p_start[:2] + s * (p_target[:2] - p_start[:2])
        vel_xy = s_dot * (p_target[:2] - p_start[:2])

        # Vertical Z trajectory: bell curve with apex at tau = 0.5
        # z(tau) = z_start + (z_target - z_start) * s + step_height * sin(pi * tau)
        z_base = p_start[2] + s * (p_target[2] - p_start[2])
        z_lift = self.step_height * np.sin(np.pi * tau)
        pos_z = z_base + z_lift

        vz_base = s_dot * (p_target[2] - p_start[2])
        vz_lift = self.step_height * (np.pi / step_duration) * np.cos(np.pi * tau)
        vel_z = vz_base + vz_lift

        pos = np.array([pos_xy[0], pos_xy[1], pos_z])
        vel = np.array([vel_xy[0], vel_xy[1], vel_z])
        return pos, vel
