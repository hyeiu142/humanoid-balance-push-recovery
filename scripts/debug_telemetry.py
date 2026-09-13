"""
Robotics Telemetry & Pipeline Debugger for Unitree G1 Push Recovery.
Theo dõi toàn diện luồng điều khiển robotics 5 tầng:
  [1. SENSOR] -> [2. ESTIMATION] -> [3. PLANNER / FSM] -> [4. MPC & IK] -> [5. ACTION & REACTION]

Xuất dữ liệu ra:
  1. Terminal: In log chi tiết theo từng sự kiện và từng chu kỳ quan trọng.
  2. File CSV: 'logs/telemetry.csv' (lưu 1000Hz toàn bộ các biến trạng thái).
  3. Biểu đồ: 'telemetry_dashboard.png' (6 đồ thị phân tích chuỗi nguyên nhân - kết quả).
"""

import os
import sys
import csv
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sim.simulation_base import SimulationBase
from controllers.v2_mpc.mpc_controller import V2MPCController, SteppingMode, RecoveryState


def run_debug_session(push_force: float = 85.0, duration: float = 2.5, stepping_mode: SteppingMode = SteppingMode.SINGLE_LEG):
    sim = SimulationBase()
    ctrl = V2MPCController(sim.model, nominal_qpos=sim.nominal_qpos, stepping_mode=stepping_mode)

    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    csv_path = os.path.join(REPO_ROOT, "logs", "telemetry.csv")
    plot_path = os.path.join(REPO_ROOT, "telemetry_dashboard.png")

    state = sim.reset()
    ctrl.reset()

    # Đặt lịch lực đẩy tại t = 0.5s (phải gọi SAU khi reset)
    sim.schedule_push([push_force, 0, 0], duration=0.1, start_time=0.5)

    # Dữ liệu phục vụ vẽ biểu đồ
    log_records = []

    print("=" * 105)
    print(f"🔬 ROBOTICS TELEMETRY & DEBUG SESSION (LỰC ĐẨY: {push_force}N, CHẾ ĐỘ: {stepping_mode.value})")
    print("=" * 105)
    print(f"{'TIME':^7} | {'[1. SENSOR / ESTIMATION]':^28} | {'[2. PLANNER / FSM]':^26} | {'[3. ACTION]':^16} | {'[4. REACTION]':^14}")
    print(f"{'t (s)':^7} | {'CoM_x':>7} {'v_x':>7} {'ICP_x':>7} {'Pitch':>5} | {'FSM State':^14} {'Step':^4} {'Mode':^6} | {'u_pitch':>7} {'u_roll':>7} | {'LF':^4} {'RF':^4} {'Z':>4}")
    print("-" * 105)

    prev_fsm = None
    prev_vx = 0.0

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time", "com_x", "com_vx", "com_ax", "icp_x", "pelvis_pitch_deg", "pelvis_roll_deg", "pelvis_z",
        "fsm_state", "stepping_mode", "step_len", "swing_leg",
        "action_left_hip_p", "action_right_hip_p", "action_left_knee", "action_right_knee",
        "action_left_ankle_p", "action_right_ankle_p",
        "left_foot_contact", "right_foot_contact", "push_fx"
    ])

    total_steps = int(duration / sim.dt)
    for step in range(total_steps):
        t = sim.data.time

        # --- [1. SENSOR & STATE ESTIMATION] ---
        com_x = state.com_pos[0]
        com_vx = state.com_vel[0]
        com_ax = (com_vx - prev_vx) / sim.dt if step > 0 else 0.0
        prev_vx = com_vx
        pitch_deg = np.degrees(state.pelvis_rpy[1])
        roll_deg = np.degrees(state.pelvis_rpy[0])
        icp_x = com_x + com_vx / ctrl.lipm.omega_0

        # --- [2. PLANNER & CONTROLLER / ACTION] ---
        action = ctrl.compute_action(state, sim.dt)
        fsm_name = ctrl.fsm_state.name
        mode_name = ctrl._active_swing_mode.name if hasattr(ctrl, '_active_swing_mode') else ctrl.stepping_mode.name

        # --- [3. SIMULATION REACTION] ---
        state, push, fell = sim.step(action)
        lf_contact = state.left_foot_contact
        rf_contact = state.right_foot_contact
        pelvis_z = state.pelvis_pos[2]

        # Ghi CSV
        csv_writer.writerow([
            f"{t:.4f}", f"{com_x:.4f}", f"{com_vx:.4f}", f"{com_ax:.4f}", f"{icp_x:.4f}",
            f"{pitch_deg:.2f}", f"{roll_deg:.2f}", f"{pelvis_z:.4f}",
            fsm_name, mode_name, f"{ctrl.cur_step_len:.3f}", getattr(ctrl, 'swing_leg', 'none'),
            f"{action[0]:.4f}", f"{action[6]:.4f}", f"{action[3]:.4f}", f"{action[9]:.4f}",
            f"{action[4]:.4f}", f"{action[10]:.4f}",
            int(lf_contact), int(rf_contact), f"{push[0]:.1f}"
        ])

        log_records.append({
            "t": t, "com_x": com_x, "com_vx": com_vx, "icp_x": icp_x,
            "pitch": pitch_deg, "roll": roll_deg, "pelvis_z": pelvis_z,
            "fsm": fsm_name, "mode": mode_name, "step_len": ctrl.cur_step_len,
            "lf": lf_contact, "rf": rf_contact, "push_fx": push[0],
            "act_hip_p": action[0], "act_knee": action[3], "act_ankle_p": action[4]
        })

        # In log: In khi có sự kiện chuyển pha FSM, hoặc khi bị đẩy, hoặc định kỳ mỗi 50ms trong pha quan trọng
        state_changed = (fsm_name != prev_fsm)
        is_pushing = (push[0] > 0.1)
        important_periodic = (0.5 <= t <= 1.2 and step % 40 == 0)

        if state_changed or is_pushing or important_periodic:
            marker = "⚡" if is_pushing else ("🔄" if state_changed else "  ")
            fsm_str = fsm_name[:14]
            step_str = f"{ctrl.cur_step_len*100:.0f}cm" if fsm_name != "DOUBLE_SUPPORT" else "----"
            mode_str = "S-LEG" if mode_name == "SINGLE_LEG" else "SYNC"
            lf_str = "YES" if lf_contact else "NO "
            rf_str = "YES" if rf_contact else "NO "

            # Rút trích tín hiệu lệnh
            u_p_cmd = f"{action[0]:+.2f}"
            u_r_cmd = f"{action[5]:+.2f}"

            print(f"{marker}{t:5.3f}s | {com_x:+6.3f} {com_vx:+6.3f} {icp_x:+6.3f} {pitch_deg:+5.1f}° | {fsm_str:14s} {step_str:^5} {mode_str:^5} | {u_p_cmd:>7} {u_r_cmd:>7} | {lf_str} {rf_str} {pelvis_z:4.2f}m")

            if state_changed:
                print(f"      └──► [SỰ KIỆN CHUYỂN PHA]: Chuyển từ {prev_fsm} sang {fsm_name} tại t={t:.3f}s (ICP={icp_x:+.3f}m)")

        prev_fsm = fsm_name

        if fell:
            print(f"\n❌ [CẢNH BÁO DEBUG] ROBOT ĐÃ NGÃ tại t={t:.3f}s! Góc nghiêng thân: Pitch={pitch_deg:.1f}°, Roll={roll_deg:.1f}°")
            break

    csv_file.close()
    print("-" * 105)
    print(f"✅ Đã lưu toàn bộ dữ liệu 1000Hz vào: {csv_path}")

    # Vẽ Dashboard chẩn đoán đa tầng
    plot_telemetry_dashboard(log_records, plot_path, push_force)
    print(f"📊 Đã xuất Dashboard đồ thị chẩn đoán tại: {plot_path}")


def plot_telemetry_dashboard(records, save_path, push_force):
    times = [r["t"] for r in records]
    com_vx = [r["com_vx"] for r in records]
    icp_x = [r["icp_x"] for r in records]
    com_x = [r["com_x"] for r in records]
    pitch = [r["pitch"] for r in records]
    roll = [r["roll"] for r in records]
    pelvis_z = [r["pelvis_z"] for r in records]
    push_fx = [r["push_fx"] for r in records]
    lf = [r["lf"] for r in records]
    rf = [r["rf"] for r in records]
    hip_p = [r["act_hip_p"] for r in records]
    ankle_p = [r["act_ankle_p"] for r in records]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    # 1. Lực đẩy & Vận tốc CoM & Capture Point (Sensor -> Estimation)
    ax1 = axes[0]
    ax1.plot(times, icp_x, label="Capture Point ICP_x (m)", color="#d62728", lw=2)
    ax1.plot(times, com_vx, label="CoM Velocity v_x (m/s)", color="#ff7f0e", lw=1.8, linestyle="--")
    ax1.plot(times, com_x, label="CoM Position x (m)", color="#1f77b4", lw=1.5)
    ax1.axhline(0.11, color="red", linestyle=":", alpha=0.7, label="Ngưỡng Kích Hoạt Bước (0.11m)")
    ax1.fill_between(times, 0, np.array(push_fx)/push_force * 0.15, color="gray", alpha=0.2, label=f"Xung lực đẩy ({push_force}N)")
    ax1.set_ylabel("Không gian X (m)")
    ax1.set_title("ROBOTICS TELEMETRY DASHBOARD: CHUỖI NGUYÊN NHÂN - KẾT QUẢ", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Góc nghiêng thân Pitch & Roll (Orientation Reaction)
    ax2 = axes[1]
    ax2.plot(times, pitch, label="Pelvis Pitch (° - Nghiêng dọc)", color="#2ca02c", lw=2)
    ax2.plot(times, roll, label="Pelvis Roll (° - Nghiêng ngang)", color="#9467bd", lw=1.5)
    ax2.set_ylabel("Góc nghiêng (°)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Lệnh điều khiển góc khớp xuất ra (Action Output)
    ax3 = axes[2]
    ax3.plot(times, hip_p, label="Lệnh khớp Háng Hip Pitch (rad)", color="#17becf", lw=1.8)
    ax3.plot(times, ankle_p, label="Lệnh khớp Cổ Chân Ankle Pitch (rad)", color="#e377c2", lw=1.8)
    ax3.set_ylabel("Góc lệnh (rad)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Trạng thái tiếp xúc đất & Độ cao Pelvis (Physical Reaction)
    ax4 = axes[3]
    ax4.plot(times, pelvis_z, label="Độ cao Pelvis Z (m)", color="#333333", lw=2)
    ax4.step(times, np.array(lf) * 0.05 + 0.65, label="Chân Trái Chạm Đất (LF)", color="#1f77b4", where="post")
    ax4.step(times, np.array(rf) * 0.05 + 0.60, label="Chân Phải Chạm Đất (RF)", color="#2ca02c", where="post")
    ax4.set_xlabel("Thời gian t (giây)")
    ax4.set_ylabel("Độ cao / Tiếp xúc")
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    force = float(sys.argv[1]) if len(sys.argv) > 1 else 85.0
    run_debug_session(push_force=force)
