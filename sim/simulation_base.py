"""
SimulationBase: Core MuJoCo simulation environment for Unitree G1 humanoid.
Provides standardized reset, step, sensor state estimation, and disturbance injection.
"""

import os
from typing import Optional, Tuple
import numpy as np
import mujoco

from sim.sensors import StateEstimator, RobotState
from sim.disturbance import DisturbanceManager


class SimulationBase:
    """
    Simulation Base wrapping MuJoCo physics, state estimation, and disturbance management.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default to bundled Unitree G1 scene
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_dir, "..", "models", "unitree_g1", "scene.xml")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {model_path}")

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.estimator = StateEstimator(self.model)
        self.disturbance_mgr = DisturbanceManager(self.model)

        # Store nominal joint positions from standing keyframe
        self.nominal_qpos = self._extract_keyframe_ctrl(0)

        # Reset environment initially
        self.reset()

    def _extract_keyframe_ctrl(self, keyframe_id: int = 0) -> np.ndarray:
        """Extract actuator joint positions from keyframe."""
        ctrl = np.zeros(self.model.nu)
        if self.model.nkey > keyframe_id:
            key_qpos = self.model.key_qpos[keyframe_id]
            for i in range(self.model.nu):
                jnt_id = self.model.actuator_trnid[i, 0]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                ctrl[i] = key_qpos[qpos_adr]
        return ctrl

    def reset(self, keyframe_id: int = 0) -> RobotState:
        """
        Reset simulation to standing keyframe and clear disturbances.
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        self.disturbance_mgr.clear()
        self.estimator.reset()

        # Set default control targets to match initial posture
        for i in range(self.model.nu):
            self.data.ctrl[i] = self.nominal_qpos[i]

        # Settle forward kinematics & contacts
        mujoco.mj_forward(self.model, self.data)
        return self.estimator.update(self.data)

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[RobotState, np.ndarray, bool]:
        """
        Step simulation forward by dt.
        Args:
            action: (nu,) target positions or torques. If None, uses nominal standing posture.
        Returns:
            (state, active_push_force, is_fallen)
        """
        # 1. Apply disturbance forces for current time
        active_push = self.disturbance_mgr.step(self.data.time, self.data)

        # 2. Apply actuator controls
        if action is not None:
            np.copyto(self.data.ctrl, action)
        else:
            np.copyto(self.data.ctrl, self.nominal_qpos)

        # 3. Advance physics
        mujoco.mj_step(self.model, self.data)

        # 4. Update sensor state
        state = self.estimator.update(self.data)
        return state, active_push, state.is_fallen

    def get_state(self) -> RobotState:
        """Get current robot state."""
        return self.estimator.update(self.data)

    def apply_push(self, force: list, duration: float = 0.1, body_name: str = "pelvis"):
        """Trigger an instant push disturbance starting at current time."""
        self.disturbance_mgr.trigger_instant_push(
            force=force,
            duration=duration,
            current_time=self.data.time,
            body_name=body_name
        )

    def schedule_push(self, force: list, duration: float, start_time: float, body_name: str = "pelvis"):
        """Schedule a push to occur at a future simulation time."""
        self.disturbance_mgr.add_push(
            force=force,
            duration=duration,
            start_time=start_time,
            body_name=body_name
        )
