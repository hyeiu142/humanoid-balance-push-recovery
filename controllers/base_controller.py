"""
BaseController: Abstract base class for all humanoid balance controllers.
Ensures standardized interface for V1 (PID), V2 (MPC), V3 (RL), and V4 (Hybrid).
"""

from abc import ABC, abstractmethod
import numpy as np
import mujoco

from sim.sensors import RobotState


class BaseController(ABC):
    """
    Abstract controller interface.
    """

    def __init__(self, model: mujoco.MjModel, nominal_qpos: np.ndarray):
        self.model = model
        self.nominal_qpos = nominal_qpos.copy()
        self.nu = model.nu

    @abstractmethod
    def reset(self):
        """Reset internal controller states, integrators, filters."""
        pass

    @abstractmethod
    def compute_action(self, state: RobotState, dt: float) -> np.ndarray:
        """
        Compute control action (target joint angles or torques) given current robot state.
        Args:
            state: RobotState dataclass with IMU, CoM, CoP, joint positions/velocities.
            dt: Simulation timestep (seconds).
        Returns:
            np.ndarray of shape (nu,) with actuator control commands.
        """
        pass
