"""
Interactive 3D Simulation for Unitree G1 Humanoid Balance & Push Recovery (V1).
Run this script to visualize the humanoid standing in MuJoCo 3D viewer.

Features:
  - AUTO-DEMO MODE (ON by default): Automatically applies periodic pushes
    (30N -> 70N -> 120N -> Lateral -> Backward) so you can immediately see
    how the robot reacts and recovers using Ankle & Hip strategies!
  - MANUAL KEYBOARD CONTROLS: Click into the 3D window to push manually:
      * Arrow Keys (Up/Down/Left/Right): Push robot in that direction
      * Keys 1, 2, 3, 4: Select push strength (30N, 70N, 120N, 150N)
      * 'D' or 'd': Toggle Auto-Demo Mode ON / OFF
      * 'C' or 'c': Toggle Controller (V1 VMC PID <-> Passive Baseline)
      * 'R' or 'r': Reset robot to standing pose
      * Spacebar: Repeat last push
"""

import os
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import mujoco
import mujoco.viewer

from sim.simulation_base import SimulationBase
from controllers.v1_pid.vmc_pid_controller import VMCPIDController


class InteractiveViewerApp:
    def __init__(self):
        print("=" * 80)
        print("   UNITREE G1 HUMANOID BALANCE & PUSH RECOVERY - 3D VIEWER")
        print("=" * 80)
        print("\n💡 LƯU Ý QUAN TRỌNG:")
        print("   - Khi cửa sổ 3D hiện lên, chế độ AUTO-DEMO sẽ tự động kích hoạt sau 2 giây!")
        print("     Robot sẽ tự động bị đẩy tuần tự để bạn quan sát phản xạ thăng bằng.")
        print("   - Để tự bấm phím thủ công, hãy CLICK CHUỘT VÀO CỬA SỔ 3D rồi bấm các phím:")
        print("     * Phím MŨI TÊN (Lên/Xuống/Trái/Phải): Tác động lực đẩy")
        print("     * Phím 1, 2, 3, 4: Đổi lực đẩy (30N / 70N / 120N / 150N)")
        print("     * Phím D: Bật/Tắt chế độ Auto-Demo")
        print("     * Phím C: Bật/Tắt controller (VMC PID <-> Passive để xem robot ngã)")
        print("     * Phím R: Reset robot đứng thẳng")
        print("=" * 80)

        self.sim = SimulationBase()
        self.controller = VMCPIDController(self.sim.model, self.sim.nominal_qpos)

        # Settings
        self.use_controller = True
        self.auto_demo = True         # Auto-push demonstration ON by default
        self.push_force_mag = 70.0    # default 70 N (Medium)
        self.push_duration = 0.1      # 100 ms
        self.last_push_dir = "Forward (+X)"
        self.last_push_vec = [70.0, 0.0, 0.0]

        # Demo sequence of pushes: (delay_interval, dir_name, force_vector, description)
        self.demo_sequence = [
            (2.5, "Forward (+X)", [30.0, 0.0, 0.0], "Đẩy nhẹ 30N tới trước -> Phản xạ Ankle Strategy"),
            (3.5, "Forward (+X)", [70.0, 0.0, 0.0], "Đẩy vừa 70N tới trước -> Phản xạ Cổ chân & Hông"),
            (3.5, "Forward (+X)", [120.0, 0.0, 0.0], "Đẩy mạnh 120N tới trước -> Kích hoạt Hip Strategy gập thân"),
            (3.5, "Lateral (+Y)", [0.0, 60.0, 0.0], "Đẩy ngang 60N sang trái -> Ankle Roll giữ chân phẳng"),
            (3.5, "Backward (-X)", [-40.0, 0.0, 0.0], "Đẩy lùi 40N về sau -> Cổ chân đẩy thân đứng dậy"),
        ]
        self.demo_index = 0
        self.last_demo_time = 0.0

        # GLFW key codes
        self.GLFW_KEY_UP = 265
        self.GLFW_KEY_DOWN = 264
        self.GLFW_KEY_LEFT = 263
        self.GLFW_KEY_RIGHT = 262
        self.GLFW_KEY_SPACE = 32

    def trigger_push(self, dir_name: str, force_vec: list, label: str = ""):
        self.last_push_dir = dir_name
        self.last_push_vec = force_vec
        self.sim.apply_push(force=force_vec, duration=self.push_duration, body_name="pelvis")
        tag = f" ({label})" if label else ""
        print(f"\n⚡ [ĐẨY ROBOT] Hướng: {dir_name:14s} | Lực: {force_vec[0]:.0f}N, {force_vec[1]:.0f}N | {tag}")

    def key_callback(self, keycode: int):
        # Arrow Keys
        if keycode == self.GLFW_KEY_UP:
            self.auto_demo = False
            self.trigger_push("Forward (+X)", [self.push_force_mag, 0.0, 0.0])
        elif keycode == self.GLFW_KEY_DOWN:
            self.auto_demo = False
            self.trigger_push("Backward (-X)", [-self.push_force_mag, 0.0, 0.0])
        elif keycode == self.GLFW_KEY_LEFT:
            self.auto_demo = False
            self.trigger_push("Lateral Left (+Y)", [0.0, self.push_force_mag, 0.0])
        elif keycode == self.GLFW_KEY_RIGHT:
            self.auto_demo = False
            self.trigger_push("Lateral Right (-Y)", [0.0, -self.push_force_mag, 0.0])
        elif keycode == self.GLFW_KEY_SPACE:
            self.auto_demo = False
            self.trigger_push(self.last_push_dir, self.last_push_vec)

        # Force levels: 1, 2, 3, 4
        elif keycode in (ord("1"), ord("!")):
            self.push_force_mag = 30.0
            print(f">> Chọn mức lực 1 (Nhẹ: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("2"), ord("@")):
            self.push_force_mag = 70.0
            print(f">> Chọn mức lực 2 (Vừa: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("3"), ord("#")):
            self.push_force_mag = 120.0
            print(f">> Chọn mức lực 3 (Mạnh: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("4"), ord("$")):
            self.push_force_mag = 150.0
            print(f">> Chọn mức lực 4 (Cực hạn: {self.push_force_mag:.0f} N)")

        # Toggle Auto-Demo: D
        elif keycode in (ord("D"), ord("d")):
            self.auto_demo = not self.auto_demo
            status = "BẬT (ON)" if self.auto_demo else "TẮT (OFF - Điều khiển thủ công)"
            print(f">> Chế độ Auto-Demo: [{status}]")
            self.last_demo_time = self.sim.data.time

        # Reset: R
        elif keycode in (ord("R"), ord("r")):
            self.sim.reset()
            self.controller.reset()
            self.last_demo_time = self.sim.data.time
            print(">> Đã Reset robot về tư thế đứng chuẩn ban đầu.")

        # Toggle controller: C
        elif keycode in (ord("C"), ord("c")):
            self.use_controller = not self.use_controller
            status = "V1 VMC PID (Bật thăng bằng chủ động)" if self.use_controller else "Passive Baseline (TẮT controller)"
            print(f">> Chuyển đổi bộ điều khiển: [{status}]")

    def run(self):
        state = self.sim.reset()
        self.controller.reset()
        self.last_demo_time = self.sim.data.time

        print("\n🚀 Đang khởi động cửa sổ MuJoCo 3D Viewer...")

        last_status_print = time.time()
        fps = 60.0
        frame_dt = 1.0 / fps
        physics_substeps = int(frame_dt / self.sim.dt)  # e.g., 0.016s / 0.002s = 8 steps

        with mujoco.viewer.launch_passive(
            self.sim.model,
            self.sim.data,
            key_callback=self.key_callback
        ) as viewer:
            while viewer.is_running():
                frame_start = time.time()

                # Run physics sub-steps for this rendering frame
                for _ in range(physics_substeps):
                    # Check Auto-Demo trigger
                    if self.auto_demo:
                        interval, dir_name, f_vec, desc = self.demo_sequence[self.demo_index]
                        if self.sim.data.time - self.last_demo_time >= interval:
                            self.trigger_push(dir_name, f_vec, label=desc)
                            self.last_demo_time = self.sim.data.time
                            self.demo_index = (self.demo_index + 1) % len(self.demo_sequence)

                    # Compute controller action
                    if self.use_controller:
                        action = self.controller.compute_action(state, self.sim.dt)
                    else:
                        action = None

                    state, push, fallen = self.sim.step(action)

                # Sync graphics at 60 FPS
                viewer.sync()

                # Print status every 1.5s
                if time.time() - last_status_print >= 1.5:
                    last_status_print = time.time()
                    tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
                    ctrl_name = "V1 VMC PID" if self.use_controller else "Passive"
                    mode_name = self.controller.coordinator.current_mode.value if self.use_controller else "None"
                    stat_str = "NGÃ (FALLEN)" if fallen else "Cân bằng (OK)"
                    demo_str = f"Auto-Demo: ON (Bài {self.demo_index + 1}/{len(self.demo_sequence)})" if self.auto_demo else "Thủ công (Manual)"
                    sys.stdout.write(
                        f"\r[{ctrl_name:10s}] {stat_str:13s} | Mode: {mode_name:15s} | Độ nghiêng: {tilt_deg:4.1f}° | Pelvis Z: {state.pelvis_pos[2]:.2f}m | {demo_str}"
                    )
                    sys.stdout.flush()

                # Maintain 60 FPS pace
                elapsed = time.time() - frame_start
                sleep_time = frame_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


def main():
    app = InteractiveViewerApp()
    app.run()


if __name__ == "__main__":
    main()
