"""
Inverse Kinematics (IK) and Foot Jacobian Solver for Unitree G1 humanoid legs.
Solves target joint angles for 6-DOF legs given desired foot positions relative to pelvis.
"""

import numpy as np
import mujoco


class LegKinematicsSolver:
    """
    Inverse Kinematics solver using Damped Least Squares (DLS) Jacobian iteration
    for Unitree G1 6-DOF legs.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # Internal scratch MjData for kinematics calculations without disturbing simulation data
        self.data = mujoco.MjData(model)

        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id < 0:
            self.pelvis_id = 1

        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

        # Actuator and DOF indices for Unitree G1:
        # Left leg joints: actuators 0..5, NV dofs 6..11
        # Right leg joints: actuators 6..11, NV dofs 12..17
        self.left_actuators = np.array([0, 1, 2, 3, 4, 5])
        self.right_actuators = np.array([6, 7, 8, 9, 10, 11])

        self.left_dofs = np.array([6, 7, 8, 9, 10, 11])
        self.right_dofs = np.array([12, 13, 14, 15, 16, 17])

        # Pre-allocate Jacobians
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve_ik(
        self,
        side: str,  # 'left' or 'right'
        target_rel_pos: np.ndarray,  # desired [x, y, z] relative to pelvis
        current_full_qpos: np.ndarray,
        max_iters: int = 15,
        tol: float = 1e-4,
        damping: float = 1e-4,
    ) -> np.ndarray:
        """
        Solve IK for 6 leg joints to achieve target_rel_pos relative to pelvis.
        Returns:
            leg_qpos: (6,) joint angles for the specified leg
        """
        # Load current joint state into scratch data
        np.copyto(self.data.qpos, current_full_qpos)

        is_left = (side.lower() == "left")
        foot_id = self.left_foot_id if is_left else self.right_foot_id
        dof_indices = self.left_dofs if is_left else self.right_dofs
        actuator_indices = self.left_actuators if is_left else self.right_actuators

        # Target relative position
        p_des = np.asarray(target_rel_pos, dtype=np.float64)

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            p_curr = self.data.xpos[foot_id] - self.data.xpos[self.pelvis_id]
            err = p_des - p_curr

            if np.linalg.norm(err) < tol:
                break

            # Compute body Jacobian
            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, foot_id)
            J = self._jacp[:, dof_indices]

            # Damped Least Squares (DLS): dq = J^T * (J * J^T + lambda^2 * I)^(-1) * err
            JJT = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)

            # Update scratch qpos
            for i, act_idx in enumerate(actuator_indices):
                jnt_id = self.model.actuator_trnid[act_idx, 0]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] += dq[i]

        # Extract resulting leg joint angles
        leg_qpos = np.zeros(6)
        for i, act_idx in enumerate(actuator_indices):
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            leg_qpos[i] = self.data.qpos[qpos_adr]

        return leg_qpos
