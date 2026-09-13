"""
Footstep Planner & Capture Point FSM for Stepping Push Recovery.
Evaluates Instantaneous Capture Point (ICP) and triggers stabilizing recovery steps.
Computes relative footstep target displacements Delta_x, Delta_y.
"""

from enum import Enum
from typing import Optional
import numpy as np

from controllers.v2_mpc.lipm_model import LIPMModel


class SteppingState(Enum):
    DOUBLE_SUPPORT = "Double Support"
    STEPPING_LEFT = "Stepping Left Foot"
    STEPPING_RIGHT = "Stepping Right Foot"
    RESTORE_SETTLE = "Restore Settle"


class FootstepPlanner:
    """
    Evaluates Capture Point and coordinates reactive stabilizing recovery steps.
    """

    def __init__(
        self,
        lipm: LIPMModel,
        step_duration: float = 0.22,   # 220 ms fast stabilizing step
        step_height: float = 0.05,     # 5 cm foot clearance
        step_trigger_icp: float = 0.08, # ICP > 8 cm triggers step
    ):
        self.lipm = lipm
        self.step_duration = step_duration
        self.step_height = step_height
        self.step_trigger_icp = step_trigger_icp

        self.state = SteppingState.DOUBLE_SUPPORT
        self.step_start_time = 0.0
        self.step_length_x = 0.0
        self.step_offset_y = 0.0

        # Permanent landed offsets
        self.landed_offset_left = np.zeros(3)
        self.landed_offset_right = np.zeros(3)

    def reset(self):
        self.state = SteppingState.DOUBLE_SUPPORT
        self.step_start_time = 0.0
        self.step_length_x = 0.0
        self.step_offset_y = 0.0
        self.landed_offset_left.fill(0.0)
        self.landed_offset_right.fill(0.0)

    def is_stepping(self) -> bool:
        return self.state in (SteppingState.STEPPING_LEFT, SteppingState.STEPPING_RIGHT)

    def update(
        self,
        current_time: float,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
    ) -> tuple[SteppingState, Optional[str], float, float]:
        """
        Update stepping FSM.
        Returns:
            (current_state, swing_foot_name, cur_step_dx, cur_step_dz)
            where cur_step_dx is the relative forward/backward swing displacement,
            and cur_step_dz is the vertical lift height.
        """
        icp_x = com_pos[0] + com_vel[0] / self.lipm.omega_0

        # 1. During active swing:
        if self.is_stepping():
            elapsed = current_time - self.step_start_time
            tau = np.clip(elapsed / self.step_duration, 0.0, 1.0)
            swing_name = "left" if self.state == SteppingState.STEPPING_LEFT else "right"

            # Smooth horizontal displacement s(tau) * step_length
            s = 0.5 * (1.0 - np.cos(np.pi * tau))
            cur_dx = s * self.step_length_x

            # Vertical lift clearance
            cur_dz = self.step_height * np.sin(np.pi * tau)

            # Check Touchdown
            if elapsed >= self.step_duration:
                if swing_name == "left":
                    self.landed_offset_left[0] = self.step_length_x
                else:
                    self.landed_offset_right[0] = self.step_length_x

                self.state = SteppingState.RESTORE_SETTLE
                return self.state, swing_name, self.step_length_x, 0.0

            return self.state, swing_name, cur_dx, cur_dz

        # 2. In Settle phase:
        if self.state == SteppingState.RESTORE_SETTLE:
            return self.state, None, 0.0, 0.0

        # 3. Double Support: check if ICP exceeds stability threshold
        trigger_fwd = (icp_x > self.step_trigger_icp) or (com_vel[0] > 0.22)
        trigger_bwd = (icp_x < -0.06) or (com_vel[0] < -0.22)

        if trigger_fwd or trigger_bwd:
            # Trigger Step!
            # Swing left if vy >= 0, else right
            swing_name = "left" if com_vel[1] >= 0 else "right"
            self.state = SteppingState.STEPPING_LEFT if swing_name == "left" else SteppingState.STEPPING_RIGHT
            self.step_start_time = current_time

            if trigger_fwd:
                # Dynamic step length proportional to forward velocity
                self.step_length_x = float(np.clip(0.16 + 0.50 * max(0.0, com_vel[0]), 0.16, 0.36))
            else:
                self.step_length_x = float(np.clip(-0.12 + 0.40 * min(0.0, com_vel[0]), -0.25, -0.12))

            return self.state, swing_name, 0.0, 0.0

        return self.state, None, 0.0, 0.0
