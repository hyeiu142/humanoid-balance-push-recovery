"""
Sensors and State Estimation module for Unitree G1 Humanoid.
Provides Center of Mass (CoM), Center of Pressure (CoP), Torso Attitude (IMU),
and Support Polygon stability metrics.
"""

from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class RobotState:
    time: float
    # Center of Mass
    com_pos: np.ndarray          # [x, y, z] in world frame
    com_vel: np.ndarray          # [vx, vy, vz] in world frame
    # Pelvis / IMU
    pelvis_pos: np.ndarray       # [x, y, z]
    pelvis_quat: np.ndarray      # [w, x, y, z]
    pelvis_rpy: np.ndarray       # [roll, pitch, yaw] in radians
    pelvis_lin_vel: np.ndarray   # [vx, vy, vz] in world frame
    pelvis_ang_vel: np.ndarray   # [wx, wy, wz] (gyro) in base/world frame
    # Actuator states
    joint_pos: np.ndarray        # (nu,) actuator positions
    joint_vel: np.ndarray        # (nu,) actuator velocities
    # Foot Contacts & Ground Reactions
    left_foot_contact: bool
    right_foot_contact: bool
    total_normal_force: float    # Fz (N)
    cop: np.ndarray              # [cop_x, cop_y] in world frame
    cop_margin: float            # Distance from CoP to nearest foot boundary (m)
    is_fallen: bool              # True if robot has fallen (pelvis < 0.45m or excessive tilt)


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo quaternion [w, x, y, z] to Roll, Pitch, Yaw (ZYX convention) in radians.
    """
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class StateEstimator:
    """
    Estimates full physical state and stability indicators for Unitree G1 humanoid.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            self.pelvis_body_id = 1

        # Locate foot bodies
        self.left_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

        # Gather collision geom IDs for left and right foot
        self.left_foot_geoms = set()
        self.right_foot_geoms = set()
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            if bid == self.left_foot_body_id:
                self.left_foot_geoms.add(gid)
            elif bid == self.right_foot_body_id:
                self.right_foot_geoms.add(gid)

        # Buffer for contact force computation
        self._c_forces = np.zeros(6, dtype=np.float64)

        # Support polygon bounds in double support nominal stance
        # Foot length: x in [-0.05, 0.12], y in [-0.18, 0.18]
        self.foot_x_min = -0.05
        self.foot_x_max = 0.12
        self.foot_y_min = -0.18
        self.foot_y_max = 0.18

        # Cache for numerical CoM velocity estimation
        self._prev_com = None
        self._prev_time = None

    def reset(self):
        """Reset state estimator internal caches."""
        self._prev_com = None
        self._prev_time = None

    def update(self, data: mujoco.MjData) -> RobotState:
        """
        Compute and return current RobotState from MjData.
        """
        time = data.time

        # 1. Pelvis state
        pelvis_pos = data.xpos[self.pelvis_body_id].copy()
        pelvis_quat = data.xquat[self.pelvis_body_id].copy()
        pelvis_rpy = quat_to_euler(pelvis_quat)

        # Velocities: root is free joint (first 6 nv)
        pelvis_lin_vel = data.qvel[0:3].copy()
        pelvis_ang_vel = data.qvel[3:6].copy()

        # 2. Center of Mass
        # data.subtree_com[self.pelvis_body_id] gives CoM of full robot subtree
        com_pos = data.subtree_com[self.pelvis_body_id].copy()
        if self._prev_com is None or self._prev_time is None or time == self._prev_time:
            com_vel = np.zeros(3)
        else:
            dt = time - self._prev_time
            com_vel = (com_pos - self._prev_com) / dt
        self._prev_com = com_pos.copy()
        self._prev_time = time

        # 3. Joint angles and velocities for actuators
        nu = self.model.nu
        joint_pos = np.zeros(nu)
        joint_vel = np.zeros(nu)
        for i in range(nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            qvel_adr = self.model.jnt_dofadr[jnt_id]
            joint_pos[i] = data.qpos[qpos_adr]
            joint_vel[i] = data.qvel[qvel_adr]

        # 4. Foot contacts and Center of Pressure (CoP)
        left_contact = False
        right_contact = False
        total_fz = 0.0
        cop_x_sum = 0.0
        cop_y_sum = 0.0

        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = con.geom1, con.geom2
            is_left = (g1 in self.left_foot_geoms or g2 in self.left_foot_geoms)
            is_right = (g1 in self.right_foot_geoms or g2 in self.right_foot_geoms)

            if is_left or is_right:
                mujoco.mj_contactForce(self.model, data, i, self._c_forces)
                fn = self._c_forces[0] # normal force in contact frame
                if fn > 0.5:
                    if is_left:
                        left_contact = True
                    if is_right:
                        right_contact = True
                    total_fz += fn
                    cop_x_sum += fn * con.pos[0]
                    cop_y_sum += fn * con.pos[1]

        if total_fz > 5.0:
            cop = np.array([cop_x_sum / total_fz, cop_y_sum / total_fz])
        else:
            # If in air or no contact, default CoP to ground projection of CoM
            cop = np.array([com_pos[0], com_pos[1]])

        # 5. Stability Margin (distance to foot support boundaries)
        # Margin is positive inside polygon, negative outside
        dist_x_min = cop[0] - self.foot_x_min
        dist_x_max = self.foot_x_max - cop[0]
        dist_y_min = cop[1] - self.foot_y_min
        dist_y_max = self.foot_y_max - cop[1]
        cop_margin = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max)

        # 6. Fall Detection
        is_fallen = bool(
            pelvis_pos[2] < 0.45 or
            abs(pelvis_rpy[0]) > 0.8 or  # > 45 deg roll
            abs(pelvis_rpy[1]) > 0.8     # > 45 deg pitch
        )

        return RobotState(
            time=time,
            com_pos=com_pos,
            com_vel=com_vel,
            pelvis_pos=pelvis_pos,
            pelvis_quat=pelvis_quat,
            pelvis_rpy=pelvis_rpy,
            pelvis_lin_vel=pelvis_lin_vel,
            pelvis_ang_vel=pelvis_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            left_foot_contact=left_contact,
            right_foot_contact=right_contact,
            total_normal_force=total_fz,
            cop=cop,
            cop_margin=cop_margin,
            is_fallen=is_fallen,
        )
