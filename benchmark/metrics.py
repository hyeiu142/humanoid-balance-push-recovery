"""
Metrics calculation module for Humanoid Balance and Push Recovery.
Calculates recovery settling time, max tilt excursion, CoM & CoP margins, and fall detection.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from sim.sensors import RobotState


@dataclass
class TrialMetrics:
    controller_name: str
    direction: str
    force_magnitude: float
    duration: float
    impulse: float               # F * dt (N*s)
    survived: bool
    fall_time: Optional[float]
    max_tilt_deg: float
    final_tilt_deg: float
    settling_time: Optional[float]  # seconds to settle within 1 degree
    max_com_excursion: float       # max distance from initial CoM
    min_cop_margin: float          # minimum distance to support polygon edge (m)


class TrajectoryLogger:
    """Logs state history during a single trial for plotting and metrics evaluation."""

    def __init__(self):
        self.times: List[float] = []
        self.tilts_deg: List[float] = []
        self.com_x: List[float] = []
        self.com_y: List[float] = []
        self.cop_x: List[float] = []
        self.cop_y: List[float] = []
        self.pelvis_z: List[float] = []
        self.forces: List[float] = []
        self.cop_margins: List[float] = []

    def log(self, state: RobotState, push_force: np.ndarray):
        self.times.append(state.time)
        tilt = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.tilts_deg.append(tilt)
        self.com_x.append(state.com_pos[0])
        self.com_y.append(state.com_pos[1])
        self.cop_x.append(state.cop[0])
        self.cop_y.append(state.cop[1])
        self.pelvis_z.append(state.pelvis_pos[2])
        self.forces.append(float(np.linalg.norm(push_force)))
        self.cop_margins.append(state.cop_margin)

    def compute_metrics(
        self,
        controller_name: str,
        direction: str,
        force_mag: float,
        duration: float,
        push_start: float,
        fell: bool,
        fall_time: Optional[float] = None,
    ) -> TrialMetrics:
        times = np.array(self.times)
        tilts = np.array(self.tilts_deg)
        com_x = np.array(self.com_x)
        com_y = np.array(self.com_y)
        margins = np.array(self.cop_margins)

        push_end = push_start + duration
        impulse = force_mag * duration
        max_tilt = float(np.max(tilts)) if len(tilts) > 0 else 0.0
        final_tilt = float(tilts[-1]) if len(tilts) > 0 else 0.0

        # CoM excursion
        if len(com_x) > 0:
            init_com = np.array([com_x[0], com_y[0]])
            com_pts = np.column_stack([com_x, com_y])
            excursions = np.linalg.norm(com_pts - init_com, axis=1)
            max_com_excursion = float(np.max(excursions))
        else:
            max_com_excursion = 0.0

        min_cop_margin = float(np.min(margins)) if len(margins) > 0 else 0.0

        # Settling time: time after push_end when tilt remains <= 1.0 degree until end
        settling_time = None
        if not fell and len(times) > 0:
            post_mask = times >= push_end
            post_times = times[post_mask]
            post_tilts = tilts[post_mask]

            # Find last time tilt was outside 1.0 deg
            outside_indices = np.where(post_tilts > 1.0)[0]
            if len(outside_indices) == 0:
                settling_time = 0.0
            else:
                last_outside_idx = outside_indices[-1]
                if last_outside_idx + 1 < len(post_times):
                    settling_time = float(post_times[last_outside_idx + 1] - push_end)
                else:
                    settling_time = float(post_times[-1] - push_end)

        return TrialMetrics(
            controller_name=controller_name,
            direction=direction,
            force_magnitude=force_mag,
            duration=duration,
            impulse=impulse,
            survived=not fell,
            fall_time=fall_time,
            max_tilt_deg=max_tilt,
            final_tilt_deg=final_tilt,
            settling_time=settling_time,
            max_com_excursion=max_com_excursion,
            min_cop_margin=min_cop_margin,
        )
