"""
Interactive 3D Simulation for Unitree G1 Humanoid Balance & Push Recovery (V1).
Run this script to visualize the humanoid standing in MuJoCo 3D viewer and apply
push disturbances in real-time using keyboard controls.

Controls:
  - Arrow Keys:
      Up:    Push Forward (+X)
      Down:  Push Backward (-X)
      Left:  Push Left (+Y)
      Right: Push Right (-Y)
  - Number Keys 1-4:
      1: Mild Push (30 N)
      2: Medium Push (70 N)
      3: Strong Push (120 N)
      4: Critical Push (150 N)
  - 'R' or 'r': Reset robot to standing pose
  - 'C' or 'c': Toggle controller (V1 VMC PID <-> Passive Baseline)
  - Spacebar:   Re-apply current selected push
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
        print("=" * 75)
        print("  UNITREE G1 HUMANOID BALANCE & PUSH RECOVERY - INTERACTIVE 3D VIEWER")
        print("=" * 75)

        self.sim = SimulationBase()
        self.controller = VMCPIDController(self.sim.model, self.sim.nominal_qpos)

        # Settings
        self.use_controller = True
        self.push_force_mag = 70.0  # default 70 N (Medium)
        self.push_duration = 0.1     # 100 ms
        self.last_push_dir = "Forward (+X)"
        self.last_push_vec = [70.0, 0.0, 0.0]

        # Key mapping (GLFW key codes)
        self.GLFW_KEY_UP = 265
        self.GLFW_KEY_DOWN = 264
        self.GLFW_KEY_LEFT = 263
        self.GLFW_KEY_RIGHT = 262
        self.GLFW_KEY_SPACE = 32

    def trigger_push(self, dir_name: str, force_vec: list):
        self.last_push_dir = dir_name
        self.last_push_vec = force_vec
        self.sim.apply_push(force=force_vec, duration=self.push_duration, body_name="pelvis")
        print(f"\n>> APPLIED PUSH: {dir_name} | Force: {force_vec} for {self.push_duration}s")

    def key_callback(self, keycode: int):
        # Arrow Keys
        if keycode == self.GLFW_KEY_UP:
            self.trigger_push("Forward (+X)", [self.push_force_mag, 0.0, 0.0])
        elif keycode == self.GLFW_KEY_DOWN:
            self.trigger_push("Backward (-X)", [-self.push_force_mag, 0.0, 0.0])
        elif keycode == self.GLFW_KEY_LEFT:
            self.trigger_push("Lateral Left (+Y)", [0.0, self.push_force_mag, 0.0])
        elif keycode == self.GLFW_KEY_RIGHT:
            self.trigger_push("Lateral Right (-Y)", [0.0, -self.push_force_mag, 0.0])
        elif keycode == self.GLFW_KEY_SPACE:
            self.trigger_push(self.last_push_dir, self.last_push_vec)

        # Force levels: 1, 2, 3, 4
        elif keycode in (ord("1"), ord("!")):
            self.push_force_mag = 30.0
            print(f">> Selected Force Level: 1 (Mild: {self.push_force_mag} N)")
        elif keycode in (ord("2"), ord("@")):
            self.push_force_mag = 70.0
            print(f">> Selected Force Level: 2 (Medium: {self.push_force_mag} N)")
        elif keycode in (ord("3"), ord("#")):
            self.push_force_mag = 120.0
            print(f">> Selected Force Level: 3 (Strong: {self.push_force_mag} N)")
        elif keycode in (ord("4"), ord("$")):
            self.push_force_mag = 150.0
            print(f">> Selected Force Level: 4 (Critical: {self.push_force_mag} N)")

        # Reset: R
        elif keycode in (ord("R"), ord("r")):
            self.sim.reset()
            self.controller.reset()
            print(">> Robot RESET to nominal standing pose.")

        # Toggle controller: C
        elif keycode in (ord("C"), ord("c")):
            self.use_controller = not self.use_controller
            status = "V1 VMC PID (Active Balance)" if self.use_controller else "Passive Baseline (No Controller)"
            print(f">> Switched Controller: [{status}]")

    def run(self):
        state = self.sim.reset()
        self.controller.reset()

        print("\nKeyboard Controls:")
        print("  [Up / Down / Left / Right] : Push Robot in that direction")
        print("  [1 / 2 / 3 / 4]            : Set Force (30N / 70N / 120N / 150N)")
        print("  [Space]                    : Repeat last push")
        print("  [C]                        : Toggle V1 PID vs Passive Baseline")
        print("  [R]                        : Reset standing posture")
        print("\nLaunching MuJoCo 3D Viewer...")

        last_print_time = time.time()

        with mujoco.viewer.launch_passive(
            self.sim.model,
            self.sim.data,
            key_callback=self.key_callback
        ) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # Compute control action
                if self.use_controller:
                    action = self.controller.compute_action(state, self.sim.dt)
                else:
                    action = None

                # Advance simulation by one step
                state, push, fallen = self.sim.step(action)

                # Sync viewer
                viewer.sync()

                # Status printout every 1 second
                if time.time() - last_print_time >= 1.0:
                    last_print_time = time.time()
                    tilt_deg = np.linalg.norm(state.pelvis_rpy[:2]) * 180.0 / np.pi
                    ctrl_name = "V1 VMC PID" if self.use_controller else "Passive"
                    mode_name = self.controller.coordinator.current_mode.value if self.use_controller else "None"
                    stat_str = "FALLEN!" if fallen else "Stable"
                    sys.stdout.write(
                        f"\r[{ctrl_name:10s}] Status: {stat_str:7s} | Mode: {mode_name:15s} | Tilt: {tilt_deg:4.1f}° | Pelvis Z: {state.pelvis_pos[2]:.2f}m | Force Level: {self.push_force_mag:3.0f}N"
                    )
                    sys.stdout.flush()

                # Maintain real-time simulation pace
                time_until_next_step = self.sim.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    app = InteractiveViewerApp()
    app.run()


if __name__ == "__main__":
    main()
