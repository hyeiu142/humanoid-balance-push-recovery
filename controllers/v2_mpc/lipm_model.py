"""
Linear Inverted Pendulum Model (LIPM) and Capture Point formulation.
Provides analytical discrete-time dynamics and Instantaneous Capture Point (ICP / DCM)
for humanoid balance prediction and step planning.
"""

import numpy as np


class LIPMModel:
    """
    Linear Inverted Pendulum Model (LIPM) for humanoid CoM dynamics.
    Equation of motion: x_ddot = omega_0^2 * (x - p_zmp)
    """

    def __init__(self, z0: float = 0.693, g: float = 9.81):
        self.z0 = float(z0)
        self.g = float(g)
        self.omega_0 = np.sqrt(self.g / self.z0)  # ~3.76 rad/s for G1

    def get_discrete_matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Exact discrete-time state-space matrices A, B for state X = [pos, vel]^T and input u = zmp:
        X_{k+1} = A * X_k + B * u_k
        """
        w = self.omega_0
        cosh_wt = np.cosh(w * dt)
        sinh_wt = np.sinh(w * dt)

        A = np.array([
            [cosh_wt, (1.0 / w) * sinh_wt],
            [w * sinh_wt, cosh_wt]
        ], dtype=np.float64)

        B = np.array([
            [1.0 - cosh_wt],
            [-w * sinh_wt]
        ], dtype=np.float64)

        return A, B

    def compute_capture_point(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Compute Instantaneous Capture Point (ICP / Divergent Component of Motion - DCM):
        xi = pos + vel / omega_0
        Works for 1D, 2D (x, y), or 3D vectors.
        """
        return np.asarray(pos) + np.asarray(vel) / self.omega_0

    def step_simulation(self, pos: float, vel: float, zmp: float, dt: float) -> tuple[float, float]:
        """Advance 1D state by dt using exact LIPM solution."""
        A, B = self.get_discrete_matrices(dt)
        X = np.array([pos, vel])
        X_next = A @ X + B.ravel() * zmp
        return float(X_next[0]), float(X_next[1])
