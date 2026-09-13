"""
Footstep decision policy for push recovery.

This module owns the decision to step, the active stepping mode, the swing
leg, and the planned step length. It deliberately does not own execution
state or joint commands; those remain in the controller and IK adapter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from controllers.v2_mpc.lipm_model import LIPMModel


class SteppingMode(Enum):
    SINGLE_LEG = "Single-Leg Stepping"
    SYNC_SHUFFLE = "Synchronous Shuffle"


@dataclass(frozen=True)
class StepPlan:
    """Immutable plan for one recovery step."""

    mode: SteppingMode
    swing_leg: str
    step_length: float
    duration: float
    icp_x: float


class FootstepPlanner:
    """Choose whether and how the controller should take a recovery step."""

    def __init__(
        self,
        lipm: LIPMModel,
        step_duration: float = 0.18,
        single_leg_duration: float = 0.16,
        step_trigger_icp: float = 0.11,
        step_trigger_velocity: float = 0.28,
        single_leg_acceleration_limit: float = 5.5,
    ):
        self.lipm = lipm
        self.step_duration = float(step_duration)
        self.single_leg_duration = float(single_leg_duration)
        self.step_trigger_icp = float(step_trigger_icp)
        self.step_trigger_velocity = float(step_trigger_velocity)
        self.single_leg_acceleration_limit = float(single_leg_acceleration_limit)

    def reset(self) -> None:
        """Keep a reset hook for controller lifecycle symmetry."""

    def plan(
        self,
        *,
        current_time: float,
        com_pos_x: float,
        com_vel_x: float,
        com_acc_x: float,
        pelvis_roll: float,
        total_offset: float,
        stepping_mode: SteppingMode,
    ) -> Optional[StepPlan]:
        """Return a recovery plan, or None when in-place balance is enough."""
        if current_time < 0.10:
            return None

        icp_x = float(com_pos_x + com_vel_x / self.lipm.omega_0)
        trigger = (
            com_vel_x > self.step_trigger_velocity
            or icp_x > total_offset + self.step_trigger_icp
        )
        if not trigger:
            return None

        predicted_touchdown = icp_x * np.exp(self.lipm.omega_0 * self.step_duration)

        if (
            stepping_mode == SteppingMode.SINGLE_LEG
            and com_acc_x < self.single_leg_acceleration_limit
        ):
            return StepPlan(
                mode=SteppingMode.SINGLE_LEG,
                swing_leg="left" if pelvis_roll > 0.03 else "right",
                step_length=0.09,
                duration=self.single_leg_duration,
                icp_x=icp_x,
            )

        return StepPlan(
            mode=SteppingMode.SYNC_SHUFFLE,
            swing_leg="both",
            step_length=float(np.clip(predicted_touchdown - total_offset, 0.16, 0.26)),
            duration=self.step_duration,
            icp_x=icp_x,
        )

    def plan_secondary_shuffle(
        self,
        *,
        icp_x: float,
        foot_center_x: float,
        current_time: float,
        last_step_start_time: float,
        step_count: int,
        max_steps: int,
    ) -> Optional[StepPlan]:
        """Plan another shuffle when ICP escapes support after landing."""
        settled_long_enough = (
            current_time - (last_step_start_time + self.step_duration) > 0.12
        )
        if (
            icp_x <= foot_center_x + 0.14
            or step_count >= max_steps
            or not settled_long_enough
        ):
            return None

        rel_icp = max(0.0, icp_x - foot_center_x)
        return StepPlan(
            mode=SteppingMode.SYNC_SHUFFLE,
            swing_leg="both",
            step_length=float(
                np.clip(
                    rel_icp * np.exp(self.lipm.omega_0 * self.step_duration),
                    0.10,
                    0.18,
                )
            ),
            duration=self.step_duration,
            icp_x=icp_x,
        )
