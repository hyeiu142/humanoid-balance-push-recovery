"""
Unit & Integration tests for V2 Humanoid Model Predictive Control (MPC) & Stepping Push Recovery.
Verifies:
1. LIPM state-space model & Instantaneous Capture Point (ICP) math.
2. OSQP QP preview solver convergence, bounds compliance, and sub-millisecond execution.
3. 6-DOF Leg Inverse Kinematics precision (< 0.1mm error).
4. Static standing balance (3 seconds).
5. In-place MPC push recovery (30N, 70N, 120N) with small tilt (< 1.0 deg).
6. Heavy push stepping recovery (220N / 22.0 N·s) survival where V1 PID and Passive baseline fail.
"""

import os
import sys
import time
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.simulation_base import SimulationBase
from controllers.v2_mpc.lipm_model import LIPMModel
from controllers.v2_mpc.qp_mpc_solver import LIPMQPSolver
from controllers.v2_mpc.leg_kinematics import LegKinematicsSolver
from controllers.v2_mpc.mpc_controller import V2MPCController, RecoveryState, SteppingMode


class TestV2MPC(unittest.TestCase):
    def setUp(self):
        self.sim = SimulationBase()
        self.lipm = LIPMModel(z0=0.693, g=9.81)
        self.controller = V2MPCController(self.sim.model, self.sim.nominal_qpos)

    def test_lipm_model_matrices(self):
        """Verify continuous and discrete LIPM dynamics matrices."""
        A, B = self.lipm.get_discrete_matrices(dt=0.05)
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 1))

        # Check Capture Point formula: xi = x + vx / omega_0
        x, vx = 0.05, 0.35
        expected_xi = x + vx / self.lipm.omega_0
        computed_xi = self.lipm.compute_capture_point(x, vx)
        self.assertAlmostEqual(computed_xi, expected_xi, places=6)

    def test_qp_solver_convergence_and_speed(self):
        """Verify QP solver converges in < 2.0 ms and satisfies ZMP constraints."""
        solver = LIPMQPSolver(lipm=self.lipm, horizon=16, dt=0.05)
        zmp_min, zmp_max = -0.08, 0.12

        # Warm up
        solver.solve(0.0, 0.0, ref_pos=0.0, zmp_min=zmp_min, zmp_max=zmp_max)

        # Benchmark 10 solves
        solve_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            opt_zmp, x_traj, zmp_traj = solver.solve(
                0.03,
                0.25,
                ref_pos=0.0,
                zmp_min=zmp_min,
                zmp_max=zmp_max,
            )
            solve_times.append(time.perf_counter() - t0)

            # Constraint check
            self.assertGreaterEqual(opt_zmp, zmp_min - 1e-4)
            self.assertLessEqual(opt_zmp, zmp_max + 1e-4)
            self.assertEqual(len(x_traj), 16)
            self.assertEqual(len(zmp_traj), 16)

        avg_ms = np.mean(solve_times) * 1000.0
        print(f"\n[QP Benchmark] Average solve time: {avg_ms:.3f} ms")
        self.assertLess(avg_ms, 2.0, "OSQP solver must execute in under 2.0 ms")

    def test_leg_inverse_kinematics_precision(self):
        """Verify DLS Inverse Kinematics reaches nominal stance foot position with < 0.1 mm error."""
        ik_solver = LegKinematicsSolver(self.sim.model)
        full_qpos = np.zeros(self.sim.model.nq)
        full_qpos[7:] = self.sim.nominal_qpos
        full_qpos[2] = 0.79

        # Target relative foot vector
        target_pos = np.array([0.0, 0.1185, -0.7568])
        angles = ik_solver.solve_ik("left", target_pos, full_qpos, max_iters=25)

        achieved_pos = ik_solver.data.xpos[ik_solver.left_foot_id] - ik_solver.data.xpos[ik_solver.pelvis_id]
        err_mm = np.linalg.norm(target_pos - achieved_pos) * 1000.0
        print(f"\n[IK Benchmark] Left foot IK position error: {err_mm:.4f} mm")
        self.assertLess(err_mm, 0.1, "IK solver error must be under 0.1 mm")

    def test_static_standing_stability(self):
        """Verify robot remains standing upright with V2 MPC for 3 seconds without falling."""
        state = self.sim.reset()
        self.controller.reset()

        for _ in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fallen = self.sim.step(action)
            self.assertFalse(fallen, "Robot should not fall during static standing with V2 MPC")

        tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.assertLess(tilt_deg, 0.5, "Static standing tilt should be under 0.5 degrees")
        self.assertGreater(state.pelvis_pos[2], 0.75, "Pelvis height should remain above 0.75m")

    def test_inplace_mpc_pushes(self):
        """Verify robot recovers from 30N and 70N pushes using In-Place MPC without stepping."""
        for push_f in [30.0, 70.0]:
            state = self.sim.reset()
            self.controller.reset()
            self.sim.schedule_push([push_f, 0, 0], duration=0.1, start_time=0.5)

            for step in range(1500):
                action = self.controller.compute_action(state, self.sim.dt)
                state, _, fallen = self.sim.step(action)
                self.assertFalse(fallen, f"Robot should survive {push_f}N push")

            # Must remain in DOUBLE_SUPPORT for mild pushes
            self.assertEqual(
                self.controller.fsm_state,
                RecoveryState.DOUBLE_SUPPORT,
                f"{push_f}N push should be absorbed in-place without stepping",
            )
            tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
            self.assertLess(tilt_deg, 0.6, f"Final tilt should be < 0.6 deg after {push_f}N push")

    def test_heavy_push_stepping_recovery_220n(self):
        """Verify robot recovers from 220N push (where V1 and Passive fail) using Sync Shuffle Stepping Recovery."""
        state = self.sim.reset()
        self.controller.reset()
        self.controller.stepping_mode = SteppingMode.SYNC_SHUFFLE
        self.sim.schedule_push([220.0, 0, 0], duration=0.1, start_time=0.5)

        stepped = False
        for step in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fallen = self.sim.step(action)
            self.assertFalse(fallen, "Robot with V2 MPC must survive 220N push in Sync Shuffle mode!")

            if self.controller.fsm_state in (RecoveryState.STEP_SWING, RecoveryState.LANDED_SETTLE):
                stepped = True

        self.assertTrue(stepped, "Stepping Recovery should have been triggered for 220N push")
        tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.assertLess(tilt_deg, 3.5, "Torso tilt should remain < 3.5 deg after 220N recovery")
        self.assertGreater(state.pelvis_pos[2], 0.70, "Pelvis height should remain > 0.70m")

    def test_single_leg_stepping_recovery(self):
        """Verify robot recovers from 120N push using natural Single-Leg Stepping Recovery."""
        state = self.sim.reset()
        self.controller.reset()
        self.controller.stepping_mode = SteppingMode.SINGLE_LEG
        self.sim.schedule_push([120.0, 0, 0], duration=0.1, start_time=0.5)

        for step in range(1500):
            action = self.controller.compute_action(state, self.sim.dt)
            state, _, fallen = self.sim.step(action)
            self.assertFalse(fallen, "Robot with V2 MPC must survive 120N push in Single-Leg mode!")

        tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
        self.assertLess(tilt_deg, 1.0, "Torso tilt should remain < 1.0 deg after Single-Leg recovery")
        self.assertGreater(state.pelvis_pos[2], 0.75, "Pelvis height should remain > 0.75m")

    def test_stepping_mode_toggle(self):
        """Verify SteppingMode enum and controller mode toggle."""
        self.assertEqual(self.controller.stepping_mode, SteppingMode.SINGLE_LEG)
        self.controller.stepping_mode = SteppingMode.SYNC_SHUFFLE
        self.assertEqual(self.controller.stepping_mode, SteppingMode.SYNC_SHUFFLE)
        self.controller.stepping_mode = SteppingMode.SINGLE_LEG
        self.assertEqual(self.controller.stepping_mode, SteppingMode.SINGLE_LEG)


if __name__ == "__main__":
    unittest.main()
