"""
Quadratic Programming (QP) Model Predictive Control (MPC) Solver for LIPM.
Formulates the preview control optimization over horizon N using OSQP with warm-starting.
Optimizes ZMP trajectory to drive CoM smoothly to desired reference while respecting support limits.
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp
import osqp

from controllers.v2_mpc.lipm_model import LIPMModel


class LIPMQPSolver:
    """
    Solves discrete-time preview MPC for 1D LIPM using OSQP:
    min_{U} 0.5 U^T H U + g^T U   s.t.  u_min <= U <= u_max
    """

    def __init__(
        self,
        lipm: LIPMModel,
        horizon: int = 16,
        dt: float = 0.05,
        qx: float = 100.0,       # CoM position tracking weight
        qv: float = 10.0,        # CoM velocity damping weight
        r_zmp: float = 1.0,      # ZMP centering weight
        r_rate: float = 0.1,     # ZMP jerk / rate smoothness weight
    ):
        self.lipm = lipm
        self.N = horizon
        self.dt = dt
        self.qx = qx
        self.qv = qv
        self.r_zmp = r_zmp
        self.r_rate = r_rate

        self.A, self.B = self.lipm.get_discrete_matrices(self.dt)

        # Precompute prediction matrices P_xs and P_xu
        # X_vec = P_xs * X_0 + P_xu * U
        # X_vec has dimension 2*N (interleaved pos, vel)
        self.P_xs = np.zeros((2 * self.N, 2))
        self.P_xu = np.zeros((2 * self.N, self.N))

        A_pow = np.eye(2)
        for i in range(self.N):
            A_pow = A_pow @ self.A
            self.P_xs[2 * i : 2 * i + 2, :] = A_pow

            for j in range(i + 1):
                # A^(i-j) * B
                A_diff = np.linalg.matrix_power(self.A, i - j)
                self.P_xu[2 * i : 2 * i + 2, j] = (A_diff @ self.B).ravel()

        # State cost matrix Q_diag (2*N x 2*N)
        Q_diag = np.zeros(2 * self.N)
        for i in range(self.N):
            Q_diag[2 * i] = self.qx
            Q_diag[2 * i + 1] = self.qv
        self.Q_mat = np.diag(Q_diag)

        # Rate difference matrix D: D * U = [u_0 - u_prev, u_1 - u_0, ...]
        D = np.eye(self.N)
        for i in range(1, self.N):
            D[i, i - 1] = -1.0
        self.D_mat = D

        # Hessian matrix H = 2 * (P_xu^T * Q * P_xu + R * I + R_rate * D^T * D)
        H_dense = 2.0 * (
            self.P_xu.T @ self.Q_mat @ self.P_xu
            + self.r_zmp * np.eye(self.N)
            + self.r_rate * (self.D_mat.T @ self.D_mat)
        )
        self.P_sparse = sp.csc_matrix(H_dense)

        # Setup OSQP problem instance
        self.prob: Optional[osqp.OSQP] = None
        self.last_u_opt = np.zeros(self.N)
        self.prev_zmp = 0.0

    def reset(self):
        """Reset solver internal states."""
        self.last_u_opt.fill(0.0)
        self.prev_zmp = 0.0

    def solve(
        self,
        current_pos: float,
        current_vel: float,
        ref_pos: float = 0.0,
        zmp_min: float = -0.05,
        zmp_max: float = 0.12,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Solve QP optimization given current state and ZMP bounds.
        Returns:
            optimal_zmp_now: first control action p_zmp[0]
            predicted_com_pos: (N,) predicted positions
            predicted_zmp: (N,) optimal ZMP sequence
        """
        X0 = np.array([current_pos, current_vel])

        # Reference vector
        X_ref = np.zeros(2 * self.N)
        for i in range(self.N):
            X_ref[2 * i] = ref_pos
            X_ref[2 * i + 1] = 0.0

        # Linear cost term g:
        # g = 2 * P_xu^T * Q * (P_xs * X_0 - X_ref) - 2 * R * U_ref - 2 * R_rate * D^T * [u_prev, 0, 0...]
        err_free = self.P_xs @ X0 - X_ref
        g = 2.0 * (self.P_xu.T @ self.Q_mat @ err_free)
        g -= 2.0 * self.r_zmp * ref_pos

        # Previous ZMP rate correction
        d0 = np.zeros(self.N)
        d0[0] = -self.prev_zmp
        g += 2.0 * self.r_rate * (self.D_mat.T @ d0)

        # Bounds: l <= I * U <= u
        l_bounds = np.full(self.N, zmp_min)
        u_bounds = np.full(self.N, zmp_max)

        A_constraints = sp.csc_matrix(np.eye(self.N))

        if self.prob is None:
            self.prob = osqp.OSQP()
            self.prob.setup(
                self.P_sparse,
                g,
                A_constraints,
                l_bounds,
                u_bounds,
                verbose=False,
                warm_start=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=400,
            )
        else:
            self.prob.update(q=g, l=l_bounds, u=u_bounds)

        res = self.prob.solve()

        if res.info.status_val in [1, 2]:  # solved or solved inaccurate
            u_opt = res.x
            self.last_u_opt = u_opt.copy()
        else:
            # Fallback to clamped reference
            u_opt = self.last_u_opt

        optimal_zmp = float(u_opt[0])
        self.prev_zmp = optimal_zmp

        # Predicted state trajectory
        X_pred = self.P_xs @ X0 + self.P_xu @ u_opt
        pred_pos = X_pred[0::2]

        return optimal_zmp, pred_pos, u_opt
