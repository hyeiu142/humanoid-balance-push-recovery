"""
Disturbance injector module.
Applies external forces / impulses to the humanoid's pelvis or torso
to simulate pushes from arbitrary directions and durations.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import mujoco


class PushImpulse:
    def __init__(self, force: np.ndarray, duration: float, start_time: float, body_name: str = "pelvis"):
        self.force = np.asarray(force, dtype=np.float64) # [Fx, Fy, Fz]
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.end_time = self.start_time + self.duration
        self.body_name = body_name

    def is_active(self, current_time: float) -> bool:
        return self.start_time <= current_time < self.end_time

    def is_finished(self, current_time: float) -> bool:
        return current_time >= self.end_time


class DisturbanceManager:
    """
    Manages timed and interactive external force disturbances applied to robot bodies.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.impulses: List[PushImpulse] = []
        self.active_force = np.zeros(3)
        self.active_body_id = -1

    def add_push(self, force: List[float], duration: float, start_time: float, body_name: str = "pelvis"):
        """Schedule a timed push."""
        self.impulses.append(PushImpulse(np.array(force), duration, start_time, body_name))

    def trigger_instant_push(self, force: List[float], duration: float, current_time: float, body_name: str = "pelvis"):
        """Instantly apply a push starting right now."""
        self.impulses.append(PushImpulse(np.array(force), duration, current_time, body_name))

    def clear(self):
        """Clear all pending and active pushes."""
        self.impulses.clear()
        self.active_force.fill(0.0)

    def step(self, current_time: float, data: mujoco.MjData) -> np.ndarray:
        """
        Apply active disturbances to data.xfrc_applied for this simulation step.
        Returns total active external force vector [Fx, Fy, Fz] applied.
        """
        # Reset applied external forces
        data.xfrc_applied.fill(0.0)
        self.active_force.fill(0.0)

        # Filter out finished impulses
        self.impulses = [imp for imp in self.impulses if not imp.is_finished(current_time)]

        for imp in self.impulses:
            if imp.is_active(current_time):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, imp.body_name)
                if bid < 0:
                    bid = 1 # default to pelvis
                # data.xfrc_applied has shape (nbody, 6): [fx, fy, fz, tx, ty, tz]
                data.xfrc_applied[bid, 0:3] += imp.force
                self.active_force += imp.force
                self.active_body_id = bid

        return self.active_force.copy()
