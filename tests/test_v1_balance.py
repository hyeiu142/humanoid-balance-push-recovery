"""
Unit & Integration tests for V1 Humanoid Balance & Push Recovery.
Verifies:
1. Static standing balance (3 seconds).
2. Mild push recovery via Ankle Strategy (30N).
3. Medium push recovery via Hip Strategy (70N & 120N).
4. Physical tipping threshold (>180N).
5. Sensor & CoP computation validity.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.simulation_base import SimulationBase
from controllers.v1_pid.vmc_pid_controller import VMCPIDController


class TestV1Balance(unittest.TestCase):
    def setUp(self):
        self.sim = SimulationBase()
        self.controller = VMCPIDController(self.sim.model, self.sim.nominal_qpos)

    def test_sensors_and_cop(self):
        """Verify normal force matches gravity and CoP is within foot bounds."""
        state = self.sim.reset()
        # Settle for 100 steps
        for _ in range(100):
            state, _, _ = self.sim.step()

        expected_gravity = sum(self.sim.model.body_mass) * 9.81
        self.assertAlmostEqual(state.total_normal_force, expected_gravity, delta=15.0)
        self.assertTrue(state.left_foot_contact)
        self.assertTrue(state.right_foot_contact)
        self.assertGreater(state.cop_margin, 0.02)
        self.assertFalse(state.is_fallen)

    def test_static_standing_stability(self):
        """Verify robot remains standing upright for 3 seconds without falling."""
        state = self.sim.reset()
        self.controller.reset()
        for _ in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fallen = self.sim.step(action)
            self.assertFalse(fallen, "Robot should not fall during static standing")

        tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.assertLess(tilt_deg, 0.5, "Static standing tilt should be under 0.5 degrees")
        self.assertGreater(state.pelvis_pos[2], 0.75, "Pelvis height should remain above 0.75m")

    def test_mild_push_ankle_recovery(self):
        """Verify robot recovers from 30N forward push via Ankle Strategy."""
        state = self.sim.reset()
        self.controller.reset()
        self.sim.schedule_push([30.0, 0, 0], duration=0.1, start_time=0.5)

        for step in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fallen = self.sim.step(action)
            self.assertFalse(fallen, "Robot should survive 30N push")

        tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.assertLess(tilt_deg, 0.5, "Robot should recover to within 0.5 deg after 30N push")

    def test_medium_push_hip_recovery(self):
        """Verify robot recovers from 120N forward push (where passive baseline falls)."""
        state = self.sim.reset()
        self.controller.reset()
        self.sim.schedule_push([120.0, 0, 0], duration=0.1, start_time=0.5)

        for step in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fallen = self.sim.step(action)
            self.assertFalse(fallen, "Robot with VMC PID should survive 120N push")

        tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.assertLess(tilt_deg, 1.0, "Robot should recover to within 1.0 deg after 120N push")

    def test_physical_tipping_limit(self):
        """Verify robot falls when pushed beyond physical polygon support limits (>180N)."""
        state = self.sim.reset()
        self.controller.reset()
        self.sim.schedule_push([200.0, 0, 0], duration=0.1, start_time=0.5)

        fell = False
        for step in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fell = self.sim.step(action)
            if fell:
                break
        self.assertTrue(fell, "Robot should fall under 200N push without stepping")


if __name__ == "__main__":
    unittest.main()
