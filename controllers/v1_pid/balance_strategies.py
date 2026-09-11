"""
Balance Strategies Coordinator for Humanoid Balance Recovery.
Implements the biomechanical Ankle Strategy and Hip Strategy logic based on
tilt angle, CoM velocity, and CoP support polygon margin.
"""

from enum import Enum
import numpy as np


class BalanceStrategyMode(Enum):
    NOMINAL = "Nominal Stance"
    ANKLE = "Ankle Strategy"
    HIP = "Hip Strategy"
    TIPPING = "Tipping / Extreme"


class StrategyCoordinator:
    """
    Monitors stability margins and computes blending weights for Ankle vs Hip strategies.
    """

    def __init__(
        self,
        ankle_pitch_threshold: float = 0.05,  # ~3 degrees
        hip_pitch_threshold: float = 0.10,    # ~6 degrees
        cop_margin_threshold: float = 0.015,  # 1.5 cm from edge
    ):
        self.ankle_pitch_threshold = ankle_pitch_threshold
        self.hip_pitch_threshold = hip_pitch_threshold
        self.cop_margin_threshold = cop_margin_threshold

        self.current_mode = BalanceStrategyMode.NOMINAL
        self.ankle_weight = 1.0
        self.hip_weight = 0.0

    def evaluate(self, tilt_error: float, tilt_vel: float, cop_margin: float) -> BalanceStrategyMode:
        """
        Evaluate stability state and determine blending weights between ankle and hip.
        """
        # Composite disturbance metric combining tilt and angular velocity
        # D = |theta| + 0.15 * |omega|
        dist = abs(tilt_error) + 0.15 * abs(tilt_vel)

        if dist < 0.02 and cop_margin > self.cop_margin_threshold:
            self.current_mode = BalanceStrategyMode.NOMINAL
            self.ankle_weight = 1.0
            self.hip_weight = 0.0
        elif dist < self.ankle_pitch_threshold and cop_margin > 0.005:
            self.current_mode = BalanceStrategyMode.ANKLE
            self.ankle_weight = 1.0
            self.hip_weight = 0.0
        elif dist < self.hip_pitch_threshold or cop_margin <= 0.005:
            self.current_mode = BalanceStrategyMode.HIP
            # Gradually blend in hip strategy as ankle approaches limit
            # smoothly transition hip_weight from 0.0 to 1.0
            t = np.clip((dist - self.ankle_pitch_threshold) / (self.hip_pitch_threshold - self.ankle_pitch_threshold), 0.0, 1.0)
            self.ankle_weight = 1.0
            self.hip_weight = t
        else:
            self.current_mode = BalanceStrategyMode.TIPPING
            self.ankle_weight = 1.0
            self.hip_weight = 1.0

        return self.current_mode
