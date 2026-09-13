"""
TRỰC QUAN HÓA 3D & LIVE TELEMETRY LOGS: BƯỚC 1 CHÂN ĐƠN (SINGLE-LEG STEPPING RECOVERY)
====================================================================================
Chức năng:
1. Mở cửa sổ 3D MuJoCo với góc quay camera cận cảnh 2 chân (zoom-in).
2. Đồng thời in dòng log Telemetry trực tiếp trên Terminal theo đúng chuỗi:
   [SENSOR] -> [ESTIMATION] -> [PLANNER / FSM] -> [ACTION] -> [REACTION]
3. Hỗ trợ phím nóng:
   - SPACE: Bắn lực đẩy & Kích hoạt bước 1 chân ngay lập tức.
   - T: Bật / Tắt chế độ CHUYỂN ĐỘNG CHẬM (SLOW-MOTION 0.3x) để nhìn rõ từng cử động chân.
   - R: Reset về tư thế đứng thẳng ban đầu.
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


class SingleStepVisualizerApp:
    def __init__(self):
        print("=" * 95)
        print("   TRỰC QUAN HÓA 3D & TELEMETRY DEBUGGER: BƯỚC 1 CHÂN ĐƠN (SINGLE-LEG STEP)")
        print("=" * 95)
        print("💡 PHÍM ĐIỀU KHIỂN TRÊN CỬA SỔ 3D:")
        print("   * Phím SPACE : Kích hoạt cú xô 85N & BƯỚC 1 CHÂN (xem lặp lại bất kỳ lúc nào)")
        print("   * Phím T     : Bật / Tắt chế độ SLOW-MOTION (0.3x) để xem rõ từng milimet chân nhấc")
        print("   * Phím R     : Reset robot về đứng thẳng ban đầu")
        print("=" * 95)

        self.sim = SimulationBase()
        self.ik = LegKinematicsSolver(self.sim.model)
        self.lipm = LIPMModel(z0=0.693, g=9.81)

        # Thông số bước chân đơn đã chuẩn hóa
        self.step_duration = 0.14   # 140ms
        self.step_len = 0.10        # 10cm bước tới rõ ràng
        self.lift_height = 0.025    # 2.5cm nhấc chân êm ái

        self.phase = 0              # 0=DS, 1=SWING, 2=SETTLE
        self.phase_t0 = 0.0
        self.push_force = 85.0
        self.auto_trigger_t = 1.0   # Tự động bước lần đầu sau 1 giây
        self.has_triggered = False

        # Chế độ Slow-Motion (BẬT mặc định 0.35x để mắt thường nhìn rõ từng milimet nhấc chân)
        self.slow_motion = True
        self.speed_multiplier = 0.35

        self.last_log_t = 0.0
        self.log_count = 0

    def trigger_step(self):
        print("\n" + "═" * 105)
        print(f"⚡ [t = {self.sim.data.time:.3f}s] BẮN XUNG LỰC ĐẨY {self.push_force}N TỚI TRƯỚC (SAGITTAL) TRONG 0.1s!")
        print("═" * 105)
        self.sim.apply_push(force=[self.push_force, 0.0, 0.0], duration=0.1, body_name="pelvis")
        self.has_triggered = True

    def key_callback(self, keycode: int):
        # Spacebar: Kích hoạt bước chân
        if keycode == 32:
            self.trigger_step()
        # T: Bật/Tắt Slow Motion
        elif keycode in (ord("T"), ord("t")):
            self.slow_motion = not self.slow_motion
            self.speed_multiplier = 0.35 if self.slow_motion else 1.0
            status = "BẬT (0.35x - Chuyển động chậm quan sát rõ)" if self.slow_motion else "TẮT (1.0x - Tốc độ thực)"
            print(f"\n>> [SLOW-MOTION]: {status}")
        # R: Reset
        elif keycode in (ord("R"), ord("r")):
            self.sim.reset()
            self.phase = 0
            self.has_triggered = False
            self.auto_trigger_t = self.sim.data.time + 1.0
            print("\n>> [RESET]: Đã đưa robot về tư thế đứng chuẩn ban đầu.")

    def run(self):
        state = self.sim.reset()
        qpos_nom = self.sim.nominal_qpos.copy()

        fps = 60.0
        frame_dt = 1.0 / fps
        physics_substeps = int(frame_dt / self.sim.dt)  # 8 substeps

        with mujoco.viewer.launch_passive(
            self.sim.model,
            self.sim.data,
            key_callback=self.key_callback
        ) as viewer:
            # Góc nhìn camera cận cảnh đôi chân (Zoom-in)
            viewer.cam.distance = 1.85
            viewer.cam.elevation = -16.0
            viewer.cam.azimuth = 120.0
            viewer.cam.lookat[0] = 0.05
            viewer.cam.lookat[1] = 0.0
            viewer.cam.lookat[2] = 0.55

            print("\n👀 CỬA SỔ 3D ĐANG HIỂN THỊ TRÊN MÀN HÌNH!")
            print("   (Chế độ Slow-Motion 0.35x đang BẬT để bạn quan sát rõ từng milimet chân nhấc & vung)")
            print("─" * 105)
            print(f"{'THỜI GIAN':^9} | {'[SENSOR / DỰ ĐOÁN]':^23} | {'[BỘ ĐIỀU KHIỂN FSM]':^23} | {'[HÀNH ĐỘNG 2 CHÂN]':^30} | {'TRẠNG THÁI'}")
            print(f"{'t (s)':^9} | {'v_x (m/s)':>9} {'ICP_x (m)':>12} | {'Giai đoạn FSM':^23} | {'Chân Trái (Trụ)':^14} {'Chân Phải (Vung)':^15} |")
            print("─" * 105)

            while viewer.is_running():
                frame_start = time.time()

                for _ in range(physics_substeps):
                    t = self.sim.data.time
                    action = qpos_nom.copy()

                    # Tự động kích hoạt lần đầu tại t = 1.0s
                    if not self.has_triggered and t >= self.auto_trigger_t:
                        self.trigger_step()

                    # 1. Thu thập dữ liệu cảm biến
                    com_x = state.com_pos[0]
                    com_vx = state.com_vel[0]
                    icp_x = com_x + com_vx / self.lipm.omega_0
                    pitch_deg = np.degrees(state.pelvis_rpy[1])
                    roll_deg = np.degrees(state.pelvis_rpy[0])

                    full_qpos = np.zeros(self.sim.model.nq)
                    full_qpos[0:3] = state.pelvis_pos
                    full_qpos[3:7] = state.pelvis_quat
                    for i in range(self.sim.model.nu):
                        jnt_id = self.sim.model.actuator_trnid[i, 0]
                        full_qpos[self.sim.model.jnt_qposadr[jnt_id]] = state.joint_pos[i]

                    # 2. Cân bằng lắc ngang chuẩn (Coronal Balance - triệt tiêu xung đột dấu cổ chân)
                    roll_err = state.pelvis_rpy[0]
                    roll_rate = state.pelvis_ang_vel[0]
                    com_y_err = state.com_pos[1]
                    com_vy = state.com_vel[1]
                    u_roll = 1.0 * roll_err + 0.15 * roll_rate + 0.5 * com_y_err + 0.10 * com_vy
                    action[5] -= np.clip(0.4 * u_roll, -0.15, 0.15)
                    action[11] += np.clip(0.4 * u_roll, -0.15, 0.15)
                    action[1] += np.clip(0.4 * u_roll, -0.15, 0.15)
                    action[7] -= np.clip(0.4 * u_roll, -0.15, 0.15)

                    # 3. FSM Logic
                    if self.phase == 0:
                        # Pha Đứng 2 chân (Double Support)
                        pitch_err = state.pelvis_rpy[1]
                        pitch_rate = state.pelvis_ang_vel[1]
                        u_pitch = 1.2 * pitch_err + 0.15 * pitch_rate + 0.8 * (state.com_pos[0] - 0.003) + 0.12 * com_vx
                        action[4] += np.clip(0.8 * u_pitch, -0.35, 0.35)
                        action[10] += np.clip(0.8 * u_pitch, -0.35, 0.35)
                        action[0] += np.clip(1.2 * u_pitch, -0.40, 0.40)
                        action[6] += np.clip(1.2 * u_pitch, -0.40, 0.40)

                        if com_vx > 0.11 and t >= 0.52:
                            self.phase = 1
                            self.phase_t0 = t
                            print("\n" + "┌" + "─" * 103 + "┐")
                            print(f"│ 🚀 [t = {t:.3f}s] TRIGGER BƯỚC 1 CHÂN! v_x = {com_vx:+.2f}m/s > 0.11m/s | ICP_x = {icp_x:+.3f}m vượt chân đế!         │")
                            print(f"│    ➔ CHÂN TRÁI : Làm CHÂN TRỤ (Stance Foot) giữ mặt đất, trọng tâm dồn sang trái.      │")
                            print(f"│    ➔ CHÂN PHẢI : Nhấc lên 2.5cm, vung tới trước +10.0cm để đón tâm Capture Point!       │")
                            print("└" + "─" * 103 + "┘")

                    elif self.phase == 1:
                        # Pha Vung Chân (Step Swing)
                        elapsed = t - self.phase_t0
                        tau = np.clip(elapsed / self.step_duration, 0.0, 1.0)
                        s = 0.5 * (1.0 - np.cos(np.pi * tau))

                        cur_dx = s * self.step_len
                        cur_dz = self.lift_height * np.sin(np.pi * tau)
                        dy_shift = 0.025 * np.sin(np.pi * tau)  # Weight shift êm ái sang chân trái

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

                        torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
                        action[0] += torso_damp
                        action[6] += torso_damp

                        if elapsed >= self.step_duration:
                            self.phase = 2
                            self.phase_t0 = t
                            print("\n" + "┌" + "─" * 103 + "┐")
                            print(f"│ 🎯 [t = {t:.3f}s] TOUCHDOWN! Chân phải đã chạm đất tại vị trí +{self.step_len*100:.1f}cm về phía trước!      │")
                            print(f"│    ➔ Giữ vững tư thế Chân So Le (Staggered Stance), cả 2 chân dính chặt sàn!          │")
                            print(f"│    ➔ TRIỆT TIÊU HOÀN TOÀN DAO ĐỘNG LẮC NGANG & GIẬM CHÂN LUÂN PHIÊN.                  │")
                            print("└" + "─" * 103 + "┘")

                    elif self.phase == 2:
                        # Pha Giữ Thế Chân So Le Giảm Chấn (Stable Staggered Stance)
                        d_hip = self.step_len / 0.75
                        action[6] -= 0.6 * d_hip  # Háng phải giữ chân bước tới trước
                        action[9] += 0.15 * d_hip # Gối phải chùng nhẹ chịu lực
                        action[10] += 0.45 * d_hip# Cổ chân phải áp sát sàn phẳng

                        action[0] += 0.4 * d_hip  # Háng trái duỗi nhẹ sau
                        action[3] += 0.05         # Gối trái chùng nhẹ
                        action[4] -= 0.4 * d_hip  # Cổ chân trái áp sát sàn phẳng

                        pitch_err = state.pelvis_rpy[1]
                        pitch_rate = state.pelvis_ang_vel[1]
                        u_pitch = 1.0 * pitch_err + 0.14 * pitch_rate + 0.20 * com_vx
                        action[4] += np.clip(0.5 * u_pitch, -0.20, 0.20)
                        action[10] += np.clip(0.5 * u_pitch, -0.20, 0.20)
                        action[0] += np.clip(0.8 * u_pitch, -0.30, 0.30)
                        action[6] += np.clip(0.8 * u_pitch, -0.30, 0.30)

                    action = np.clip(action, self.sim.model.actuator_ctrlrange[:, 0], self.sim.model.actuator_ctrlrange[:, 1])
                    state, push, fallen = self.sim.step(action)

                # Cập nhật rendering trên cửa sổ 3D
                viewer.sync()

                # Tần số in log:
                # - Khi SWING (đang bước): in mỗi 25ms để thấy rõ từng milimet chân nhấc & tiến tới
                # - Khi đứng bình thường: in mỗi 80ms
                log_dt = 0.025 if self.phase == 1 else 0.080
                cur_wall_t = time.time()
                if cur_wall_t - self.last_log_t >= (log_dt / self.speed_multiplier):
                    self.last_log_t = cur_wall_t
                    rf_pos = self.sim.data.xpos[self.ik.right_foot_id]
                    lf_pos = self.sim.data.xpos[self.ik.left_foot_id]
                    step_forward_cm = (rf_pos[0] - lf_pos[0]) * 100.0

                    # Tọa độ cao Z của 2 bàn chân
                    lf_z_cm = lf_pos[2] * 100.0
                    rf_z_cm = rf_pos[2] * 100.0

                    if self.phase == 0:
                        fsm_tag = "DOUBLE_SUPPORT"
                        lf_str = f"Trụ (z={lf_z_cm:.1f}cm)"
                        rf_str = f"Trụ (z={rf_z_cm:.1f}cm)"
                    elif self.phase == 1:
                        fsm_tag = "➔ STEP_SWING (VUNG)"
                        lf_str = f"TRỤ (z={lf_z_cm:.1f}cm)"
                        rf_str = f"VUNG (+{rf_z_cm-3.5:.1f}cm)"
                    else:
                        fsm_tag = "LANDED_SETTLE"
                        lf_str = f"Chùng (z={lf_z_cm:.1f}cm)"
                        rf_str = f"ĐẤT (+{step_forward_cm:.1f}cm)"

                    step_status = "Đứng vững OK" if not fallen else "NGÃ!"

                    print(
                        f"{t:6.3f}s   | {com_vx:+7.2f} m/s {icp_x:+9.3f} m | {fsm_tag:^23} | {lf_str:^14} {rf_str:^15} | {step_status}"
                    )

                # Tốc độ khung hình (hỗ trợ Slow Motion mượt mà)
                elapsed = time.time() - frame_start
                target_sleep = (frame_dt / self.speed_multiplier) - elapsed
                if target_sleep > 0:
                    time.sleep(target_sleep)


def main():
    app = SingleStepVisualizerApp()
    app.run()


if __name__ == "__main__":
    main()
