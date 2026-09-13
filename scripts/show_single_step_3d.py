"""
Trực quan hóa 3D: Bước 1 chân đơn (Single-Leg Stepping Recovery) trên Robot Unitree G1.

Mở cửa sổ 3D MuJoCo trực tiếp trên màn hình:
- t = 0..1.0s: Robot đứng thăng bằng 2 chân.
- t = 1.0s: Bắn lực xô 85N về phía trước.
- t = 1.08s: Robot nhấc chân phải lên, nghiêng nhẹ trọng tâm sang chân trái làm trụ.
- t = 1.20s: Chân phải bước ra phía trước 8-10cm, tiếp đất thành dáng chân so le vững chắc!
- Phím SPACE: Bấm để lặp lại cú bước 1 chân bất cứ lúc nào.
- Phím R: Reset về tư thế đứng ban đầu.
"""

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import mujoco
import mujoco.viewer

from sim.simulation_base import SimulationBase
from controllers.v2_mpc.leg_kinematics import LegKinematicsSolver
from controllers.v2_mpc.lipm_model import LIPMModel


class SingleStep3DVisualizer:
    def __init__(self):
        print("=" * 80)
        print("   TRỰC QUAN HÓA 3D: BƯỚC 1 CHÂN ĐƠN (SINGLE-LEG STEPPING RECOVERY)")
        print("=" * 80)
        print("💡 HƯỚNG DẪN:")
        print("   - Cửa sổ 3D sẽ tự động xô robot và bước 1 chân sau 1.2 giây.")
        print("   - Bấm phím SPACE trên cửa sổ 3D để xem lại cú bước 1 chân.")
        print("   - Bấm phím R để Reset về đứng thẳng.")
        print("=" * 80)

        self.sim = SimulationBase()
        self.ik = LegKinematicsSolver(self.sim.model)
        self.lipm = LIPMModel(z0=0.693, g=9.81)

        self.step_duration = 0.13
        self.step_len = 0.09
        self.phase = 0  # 0: DS, 1: SWING, 2: SETTLE
        self.phase_t0 = 0.0

        self.trigger_time = 1.2  # Auto-trigger at 1.2s
        self.push_applied = False
        self.push_force = 85.0

    def trigger_step(self):
        print("\n⚡ [LỆNH] BẮT ĐẦU KỊCH BẢN BƯỚC 1 CHÂN!")
        self.sim.apply_push(force=[self.push_force, 0.0, 0.0], duration=0.1, body_name="pelvis")
        self.push_applied = True
        self.phase = 0

    def key_callback(self, keycode: int):
        # Spacebar (32): Trigger single step
        if keycode == 32:
            self.trigger_step()
        # R (Reset):
        elif keycode in (ord("R"), ord("r")):
            self.sim.reset()
            self.phase = 0
            self.push_applied = False
            self.trigger_time = self.sim.data.time + 1.0
            print(">> Đã Reset về tư thế đứng ban đầu.")

    def run(self):
        state = self.sim.reset()
        qpos_nom = self.sim.nominal_qpos.copy()

        fps = 60.0
        frame_dt = 1.0 / fps
        physics_substeps = int(frame_dt / self.sim.dt)  # 8 substeps per frame

        with mujoco.viewer.launch_passive(
            self.sim.model,
            self.sim.data,
            key_callback=self.key_callback
        ) as viewer:
            # Set initial camera view to clearly see the legs
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -12.0
            viewer.cam.azimuth = 135.0  # Angled perspective view
            viewer.cam.lookat[2] = 0.65

            print("\n👀 Cửa sổ 3D đã mở! Đang chuẩn bị trình diễn bước chân...")

            while viewer.is_running():
                frame_start = time.time()

                for _ in range(physics_substeps):
                    t = self.sim.data.time
                    action = qpos_nom.copy()

                    # Auto trigger at t = trigger_time
                    if not self.push_applied and t >= self.trigger_time:
                        self.trigger_step()

                    full_qpos = np.zeros(self.sim.model.nq)
                    full_qpos[0:3] = state.pelvis_pos
                    full_qpos[3:7] = state.pelvis_quat
                    for i in range(self.sim.model.nu):
                        jnt_id = self.sim.model.actuator_trnid[i, 0]
                        full_qpos[self.sim.model.jnt_qposadr[jnt_id]] = state.joint_pos[i]

                    # 1. Lateral Roll balance
                    roll_err = state.pelvis_rpy[0]
                    roll_rate = state.pelvis_ang_vel[0]
                    u_roll = 1.6 * roll_err + 0.20 * roll_rate
                    action[5] += np.clip(0.6 * u_roll, -0.20, 0.20)
                    action[11] += np.clip(0.6 * u_roll, -0.20, 0.20)
                    action[1] -= np.clip(0.8 * u_roll, -0.30, 0.30)
                    action[7] -= np.clip(0.8 * u_roll, -0.30, 0.30)

                    # 2. FSM Logic
                    if self.phase == 0:
                        pitch_err = state.pelvis_rpy[1]
                        pitch_rate = state.pelvis_ang_vel[1]
                        u_pitch = 1.2 * pitch_err + 0.15 * pitch_rate + 0.8 * (state.com_pos[0] - 0.003) + 0.12 * state.com_vel[0]
                        action[4] += np.clip(0.8 * u_pitch, -0.35, 0.35)
                        action[10] += np.clip(0.8 * u_pitch, -0.35, 0.35)
                        action[0] += np.clip(1.2 * u_pitch, -0.40, 0.40)
                        action[6] += np.clip(1.2 * u_pitch, -0.40, 0.40)

                        if self.push_applied and state.com_vel[0] > 0.11:
                            self.phase = 1
                            self.phase_t0 = t
                            print(f"[t = {t:.2f}s] 🦵 NHẤC CHÂN PHẢI LÊN! Chân trái làm trụ, vung chân phải tới trước...")

                    elif self.phase == 1:
                        elapsed = t - self.phase_t0
                        tau = np.clip(elapsed / self.step_duration, 0.0, 1.0)
                        s = 0.5 * (1.0 - np.cos(np.pi * tau))

                        cur_dx = s * self.step_len
                        cur_dz = 0.018 * np.sin(np.pi * tau)  # Lift foot by 1.8cm
                        dy_shift = 0.035 * np.sin(np.pi * tau)  # Lean toward left stance foot

                        t_left = np.array([0.0, 0.1185 - dy_shift, -0.7568])
                        t_right = np.array([cur_dx, -0.1185 - dy_shift, -0.7568 + cur_dz])

                        la = self.ik.solve_ik('left', t_left, full_qpos)
                        ra = self.ik.solve_ik('right', t_right, full_qpos)

                        action[0] = la[0]
                        action[3] = la[3]
                        action[4] = -(la[0] + la[3] + state.pelvis_rpy[1])

                        action[6] = ra[0]
                        action[9] = ra[3]
                        action[10] = -(ra[0] + ra[3] + state.pelvis_rpy[1])

                        action[1] = la[1]
                        action[7] = ra[1]

                        torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
                        action[0] += torso_damp
                        action[6] += torso_damp

                        if elapsed >= self.step_duration:
                            self.phase = 2
                            self.phase_t0 = t
                            print(f"[t = {t:.2f}s] 🎯 TIẾP ĐẤT! Chân phải đã đặt ở phía trước (+{self.step_len*100:.0f}cm). Robot đứng thăng bằng vững chắc!")

                    elif self.phase == 2:
                        d_hip = self.step_len / 0.75
                        action[6] -= 0.5 * d_hip
                        action[10] += 0.5 * d_hip
                        action[0] += 0.5 * d_hip
                        action[4] -= 0.5 * d_hip

                        action[3] += 0.05
                        action[9] += 0.05

                        pitch_err = state.pelvis_rpy[1]
                        pitch_rate = state.pelvis_ang_vel[1]
                        u_pitch = 1.2 * pitch_err + 0.16 * pitch_rate + 0.25 * state.com_vel[0]
                        action[4] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                        action[10] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                        action[0] += np.clip(1.0 * u_pitch, -0.35, 0.35)
                        action[6] += np.clip(1.0 * u_pitch, -0.35, 0.35)

                    action = np.clip(action, self.sim.model.actuator_ctrlrange[:, 0], self.sim.model.actuator_ctrlrange[:, 1])
                    state, push, fallen = self.sim.step(action)

                viewer.sync()

                elapsed = time.time() - frame_start
                sleep_time = frame_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


def main():
    app = SingleStep3DVisualizer()
    app.run()


if __name__ == '__main__':
    main()
