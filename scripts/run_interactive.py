"""
Interactive 3D Simulation for Unitree G1 Humanoid Balance & Push Recovery (V1 vs V2).
Run this script to visualize the humanoid standing in MuJoCo 3D viewer.

Features:
  - AUTO-DEMO MODE (ON by default): Automatically applies periodic pushes
    (30N -> 70N -> 120N -> 220N Stepping Recovery -> Lateral -> Backward) so you can immediately see
    how the robot reacts and recovers using In-Place MPC and Stepping Recovery!
  - CONTROLLER TOGGLE ('M' or 'C'):
      * V2: MPC with Stepping Recovery (Default - Highest push tolerance 220N+)
      * V1: Virtual Model Control (VMC) PID
      * Passive: Open-loop nominal stance (falls easily)
  - MANUAL KEYBOARD CONTROLS: Click into the 3D window to push manually:
      * Arrow Keys (Up/Down/Left/Right): Push robot in that direction
      * Keys 1, 2, 3, 4, 5: Select push strength (30N, 70N, 120N, 150N, 220N)
      * 'D' or 'd': Toggle Auto-Demo Mode ON / OFF
      * 'M' or 'm' / 'C' or 'c': Cycle Controller (V2 MPC <-> Passive <-> V1 PID)
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
from controllers.v2_mpc.mpc_controller import V2MPCController, SteppingMode, RecoveryState


class InteractiveViewerApp:
    def __init__(self):
        print("=" * 85)
        print("   UNITREE G1 HUMANOID BALANCE & PUSH RECOVERY - 3D INTERACTIVE VIEWER")
        print("=" * 85)
        print("\n💡 HƯỚNG DẪN ĐIỀU KHIỂN & PHÍM TẮT:")
        print("   - Khi cửa sổ 3D hiện lên, AUTO-DEMO sẽ tự động kích hoạt sau 2.5 giây.")
        print("     Robot sẽ tự động trải nghiệm các lực đẩy từ 30N đến 220N.")
        print("   - CLICK CHUỘT VÀO CỬA SỔ 3D rồi bấm các phím:")
        print("     * Phím B: THỰC HIỆN BƯỚC 1 CHÂN NGAY LẬP TỨC (Single-Leg Step)")
        print("     * Phím V: Đổi chế độ bước V2 (Single-Leg Step <-> Sync Shuffle)")
        print("     * Phím W / MŨI TÊN LÊN: Tác động lực đẩy tới trước")
        print("     * Phím 1..5: Đổi lực đẩy (1: 30N | 2: 85N | 3: 120N | 4: 150N | 5: 220N)")
        print("     * Phím D: Bật / Tắt chế độ Auto-Demo")
        print("     * Phím R: Reset robot về tư thế đứng ban đầu")
        print("     * Phím Spacebar: Đẩy lại cú đẩy gần nhất")
        print("=" * 85)

        self.sim = SimulationBase()
        self.v1_controller = VMCPIDController(self.sim.model, self.sim.nominal_qpos)
        self.v2_controller = V2MPCController(self.sim.model, self.sim.nominal_qpos)

        # Controller selection: 0: Passive, 1: V1 PID, 2: V2 MPC
        self.ctrl_modes = ["Passive Baseline", "V1: VMC PID", "V2: MPC Stepping"]
        self.current_ctrl_idx = 2  # Default to V2 MPC

        # Settings
        self.auto_demo = True         # Auto-push demonstration ON by default
        self.push_force_mag = 70.0    # default 70 N (Medium)
        self.push_duration = 0.1      # 100 ms
        self.last_push_dir = "Forward (+X)"
        self.last_push_vec = [70.0, 0.0, 0.0]

        # Demo sequence of pushes: (delay_interval, dir_name, force_vector, description)
        self.demo_sequence = [
            (2.5, "Forward (+X)", [30.0, 0.0, 0.0], "Đẩy nhẹ 30N -> In-Place Balance"),
            (3.5, "Forward (+X)", [70.0, 0.0, 0.0], "Đẩy vừa 70N -> LIPM Preview Balance"),
            (3.5, "Forward (+X)", [120.0, 0.0, 0.0], "Đẩy mạnh 120N -> Cổ chân & Hông giữ thẳng"),
            (4.0, "Forward (+X)", [220.0, 0.0, 0.0], "ĐẨY CỰC MẠNH 220N -> BƯỚC CHÂN PHỤC HỒI (V2 STEPPING)!"),
            (4.0, "Lateral (+Y)", [0.0, 60.0, 0.0], "Đẩy ngang 60N sang trái -> Ankle Roll thăng bằng"),
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

    def get_current_controller(self):
        if self.current_ctrl_idx == 1:
            return self.v1_controller
        elif self.current_ctrl_idx == 2:
            return self.v2_controller
        return None

    def trigger_push(self, dir_name: str, force_vec: list, label: str = ""):
        self.last_push_dir = dir_name
        self.last_push_vec = force_vec
        self.sim.apply_push(force=force_vec, duration=self.push_duration, body_name="pelvis")
        tag = f" ({label})" if label else ""
        print(f"\n⚡ [ĐẨY ROBOT] Hướng: {dir_name:14s} | Lực: {force_vec[0]:.0f}N, {force_vec[1]:.0f}N | {tag}")

    def cycle_controller(self):
        self.current_ctrl_idx = (self.current_ctrl_idx + 1) % 3
        name = self.ctrl_modes[self.current_ctrl_idx]
        print(f"\n🔄 [CHUYỂN CONTROLLER] Hiện tại: [{name}]")
        ctrl = self.get_current_controller()
        if ctrl is not None:
            ctrl.reset()

    def key_callback(self, keycode: int):
        # Arrow Keys & WASD
        if keycode in (self.GLFW_KEY_UP, ord("W"), ord("w")):
            self.auto_demo = False
            self.trigger_push("Forward (+X)", [self.push_force_mag, 0.0, 0.0])
        elif keycode in (self.GLFW_KEY_DOWN, ord("S"), ord("s")):
            self.auto_demo = False
            self.trigger_push("Backward (-X)", [-self.push_force_mag, 0.0, 0.0])
        elif keycode in (self.GLFW_KEY_LEFT, ord("A"), ord("a")):
            self.auto_demo = False
            self.trigger_push("Lateral Left (+Y)", [0.0, self.push_force_mag, 0.0])
        elif keycode in (self.GLFW_KEY_RIGHT, ord("E"), ord("e")):
            self.auto_demo = False
            self.trigger_push("Lateral Right (-Y)", [0.0, -self.push_force_mag, 0.0])
        elif keycode == self.GLFW_KEY_SPACE:
            self.auto_demo = False
            self.trigger_push(self.last_push_dir, self.last_push_vec)

        # Force levels: 1..5
        elif keycode in (ord("1"), ord("!")):
            self.push_force_mag = 30.0
            print(f">> Chọn mức lực 1 (Nhẹ: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("2"), ord("@")):
            self.push_force_mag = 85.0
            print(f">> Chọn mức lực 2 (Vừa - Kích hoạt Bước 1 chân: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("3"), ord("#")):
            self.push_force_mag = 120.0
            print(f">> Chọn mức lực 3 (Mạnh: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("4"), ord("$")):
            self.push_force_mag = 150.0
            print(f">> Chọn mức lực 4 (Cực hạn V1: {self.push_force_mag:.0f} N)")
        elif keycode in (ord("5"), ord("%")):
            self.push_force_mag = 220.0
            print(f">> Chọn mức lực 5 (Siêu mạnh V2 Stepping: {self.push_force_mag:.0f} N)")

        # Toggle Auto-Demo: D
        elif keycode in (ord("D"), ord("d")):
            self.auto_demo = not self.auto_demo
            status = "BẬT (ON)" if self.auto_demo else "TẮT (OFF - Điều khiển thủ công)"
            print(f">> Chế độ Auto-Demo: [{status}]")
            self.last_demo_time = self.sim.data.time

        # Reset: R
        elif keycode in (ord("R"), ord("r")):
            self.sim.reset()
            if self.v1_controller:
                self.v1_controller.reset()
            if self.v2_controller:
                self.v2_controller.reset()
            self.last_demo_time = self.sim.data.time
            print(">> Đã Reset robot về tư thế đứng chuẩn ban đầu.")

        # Toggle controller: M or C
        elif keycode in (ord("M"), ord("m"), ord("C"), ord("c")):
            self.cycle_controller()

        # Toggle V2 Stepping Mode: V or v
        elif keycode in (ord("V"), ord("v")):
            if self.v2_controller.stepping_mode == SteppingMode.SINGLE_LEG:
                self.v2_controller.stepping_mode = SteppingMode.SYNC_SHUFFLE
            else:
                self.v2_controller.stepping_mode = SteppingMode.SINGLE_LEG
            print(f">> [V2 STEPPING MODE] Chuyển sang: {self.v2_controller.stepping_mode.value}")

        # Force Single-Leg Step: B or b
        elif keycode in (ord("B"), ord("b")):
            self.auto_demo = False
            self.current_ctrl_idx = 2
            self.v2_controller.fsm_state = RecoveryState.STEP_SWING
            self.v2_controller.step_start_t = self.sim.data.time
            self.v2_controller.step_count += 1
            self.v2_controller._active_swing_mode = SteppingMode.SINGLE_LEG
            self.v2_controller.cur_step_len = 0.09
            self.v2_controller.swing_leg = "right"
            self.trigger_push("Forward (+X)", [85.0, 0.0, 0.0], label="BƯỚC 1 CHÂN (SINGLE-LEG STEP)")
            print(">> [BƯỚC 1 CHÂN] Chân phải nhấc lên vung tới trước đón đà!")

    def run(self):
        state = self.sim.reset()
        self.v1_controller.reset()
        self.v2_controller.reset()
        self.last_demo_time = self.sim.data.time

        print("\n🚀 Đang khởi động cửa sổ MuJoCo 3D Viewer...")

        last_status_print = time.time()
        fps = 60.0
        frame_dt = 1.0 / fps
        physics_substeps = int(frame_dt / self.sim.dt)  # 0.016s / 0.002s = 8 steps

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
                    ctrl = self.get_current_controller()
                    if ctrl is not None:
                        action = ctrl.compute_action(state, self.sim.dt)
                    else:
                        action = None

                    state, push, fallen = self.sim.step(action)

                # Sync graphics at 60 FPS
                viewer.sync()

                # Print status every 1.5s
                if time.time() - last_status_print >= 1.5:
                    last_status_print = time.time()
                    tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
                    ctrl_name = self.ctrl_modes[self.current_ctrl_idx]
                    
                    if self.current_ctrl_idx == 2:
                        sub_mode = f"{self.v2_controller.fsm_state.name} ({self.v2_controller.stepping_mode.name})"
                    elif self.current_ctrl_idx == 1:
                        sub_mode = self.v1_controller.coordinator.current_mode.value
                    else:
                        sub_mode = "Passive"

                    stat_str = "NGÃ (FALLEN)" if fallen else "Cân bằng (OK)"
                    demo_str = f"Auto-Demo: (Bài {self.demo_index + 1}/{len(self.demo_sequence)})" if self.auto_demo else "Manual"
                    icp_x = state.com_pos[0] + state.com_vel[0] / self.v2_controller.lipm.omega_0

                    sys.stdout.write(
                        f"\r[{ctrl_name:16s}] {stat_str:12s} | FSM: {sub_mode:25s} | Nghiêng: {tilt_deg:4.1f}° | ICP_x: {icp_x:+.2f}m | Pelvis Z: {state.pelvis_pos[2]:.2f}m | {demo_str}"
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
