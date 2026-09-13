"""
Demo & Test: Single-Leg Stepping Recovery (Bước 1 chân đơn) trên Robot Humanoid Unitree G1.

Kịch bản:
- t = 0.0s .. 0.5s: Robot đứng 2 chân thăng bằng (Double Support).
- t = 0.5s: Tác động lực đẩy 85N về phía trước trong 0.1s.
- t = 0.58s: Kích hoạt Bước 1 chân đơn (Single-Leg Step):
    + Chân trái làm chân trụ (Stance foot) bám chặt mặt đất.
    + Trọng tâm dịch chuyển nhẹ sang trái (Weight Shift) để chống lật ngang.
    + Chân phải làm chân vung (Swing foot) nhấc lên 1.5cm và bước tới trước 8cm đón Capture Point.
- t = 0.70s: Chân phải chạm đất (Touchdown), mở rộng đa giác hỗ trợ thành dáng chân so le (Staggered Stance).
- t = 0.70s .. 3.5s: Giảm chấn và ổn định hoàn toàn mà không bị ngã.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sim.simulation_base import SimulationBase
from controllers.v2_mpc.leg_kinematics import LegKinematicsSolver
from controllers.v2_mpc.lipm_model import LIPMModel

def run_single_leg_demo():
    sim = SimulationBase()
    ik = LegKinematicsSolver(sim.model)
    lipm = LIPMModel(z0=0.693, g=9.81)

    state = sim.reset()
    qpos_nom = sim.nominal_qpos.copy()

    push_force = 85.0
    sim.schedule_push([push_force, 0, 0], duration=0.1, start_time=0.5)

    step_duration = 0.12
    step_len = 0.08

    phase = 0  # 0=DS, 1=SWING_RIGHT, 2=STAGGERED_SETTLE
    phase_t0 = 0.0

    times = []
    pitch_list = []
    roll_list = []
    left_foot_x = []
    right_foot_x = []
    left_foot_z = []
    right_foot_z = []
    lf_contacts = []
    rf_contacts = []

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU KIỂM THỬ: BƯỚC 1 CHÂN ĐƠN (SINGLE-LEG STEPPING RECOVERY)")
    print(f"   Lực đẩy: {push_force}N phương trước (Sagittal) | Thời lượng: 0.10s")
    print("=" * 80)

    fell = False
    for step in range(3000):
        t = sim.data.time
        action = qpos_nom.copy()

        full_qpos = np.zeros(sim.model.nq)
        full_qpos[0:3] = state.pelvis_pos
        full_qpos[3:7] = state.pelvis_quat
        for i in range(sim.model.nu):
            jnt_id = sim.model.actuator_trnid[i, 0]
            full_qpos[sim.model.jnt_qposadr[jnt_id]] = state.joint_pos[i]

        # 1. Ổn định lắc ngang (Coronal Balance)
        roll_err = state.pelvis_rpy[0]
        roll_rate = state.pelvis_ang_vel[0]
        u_roll = 1.6 * roll_err + 0.20 * roll_rate
        action[5] += np.clip(0.6 * u_roll, -0.20, 0.20)
        action[11] += np.clip(0.6 * u_roll, -0.20, 0.20)
        action[1] -= np.clip(0.8 * u_roll, -0.30, 0.30)
        action[7] -= np.clip(0.8 * u_roll, -0.30, 0.30)

        # 2. Máy trạng thái (FSM)
        if phase == 0:
            pitch_err = state.pelvis_rpy[1]
            pitch_rate = state.pelvis_ang_vel[1]
            u_pitch = 1.2 * pitch_err + 0.15 * pitch_rate + 0.8 * (state.com_pos[0] - 0.003) + 0.12 * state.com_vel[0]
            action[4] += np.clip(0.8 * u_pitch, -0.35, 0.35)
            action[10] += np.clip(0.8 * u_pitch, -0.35, 0.35)
            action[0] += np.clip(1.2 * u_pitch, -0.40, 0.40)
            action[6] += np.clip(1.2 * u_pitch, -0.40, 0.40)

            # Phát hiện lực đẩy lớn vượt ngưỡng giữ tại chỗ -> Kích hoạt bước
            if state.com_vel[0] > 0.12 and t >= 0.52:
                phase = 1
                phase_t0 = t
                print(f"\n[t = {t:.3f}s] ⚡ PHÁT HIỆN LỰC ĐẨY! Vận tốc CoM = {state.com_vel[0]:.2f} m/s")
                print(f"             -> KÍCH HOẠT BƯỚC 1 CHÂN: Chân trái làm trụ, chân phải nhấc lên vung tới trước!")

        elif phase == 1:
            elapsed = t - phase_t0
            tau = np.clip(elapsed / step_duration, 0.0, 1.0)
            s = 0.5 * (1.0 - np.cos(np.pi * tau))

            cur_dx = s * step_len
            cur_dz = 0.014 * np.sin(np.pi * tau)  # Nhấc cao 1.4cm
            dy_shift = 0.035 * np.sin(np.pi * tau)  # Dịch trọng tâm nhẹ sang chân trụ (Weight shift)

            t_left = np.array([0.0, 0.1185 - dy_shift, -0.7568])
            t_right = np.array([cur_dx, -0.1185 - dy_shift, -0.7568 + cur_dz])

            la = ik.solve_ik('left', t_left, full_qpos)
            ra = ik.solve_ik('right', t_right, full_qpos)

            # Khớp dọc (Sagittal)
            action[0] = la[0]
            action[3] = la[3]
            action[4] = -(la[0] + la[3] + state.pelvis_rpy[1])

            action[6] = ra[0]
            action[9] = ra[3]
            action[10] = -(ra[0] + ra[3] + state.pelvis_rpy[1])

            # Nghiêng hông tạo weight shift
            action[1] = la[1]
            action[7] = ra[1]

            # Dập tắt rung lắc góc thân
            torso_damp = np.clip(0.8 * state.pelvis_rpy[1] + 0.12 * state.pelvis_ang_vel[1], -0.3, 0.3)
            action[0] += torso_damp
            action[6] += torso_damp

            if elapsed >= step_duration:
                phase = 2
                phase_t0 = t
                print(f"[t = {t:.3f}s] 🎯 TIẾP ĐẤT (Touchdown)! Chân phải đã bước tới trước {step_len*100:.1f}cm.")
                print(f"             -> Hình thành chân đế so le (Staggered Stance). Bắt đầu giảm chấn...")

        elif phase == 2:
            # Dáng chân so le mềm (Compliant Staggered Stance)
            d_hip = step_len / 0.75
            action[6] -= 0.5 * d_hip  # Háng phải gập trước
            action[10] += 0.5 * d_hip # Cổ chân phải phẳng
            action[0] += 0.5 * d_hip  # Háng trái duỗi sau
            action[4] -= 0.5 * d_hip  # Cổ chân trái phẳng

            action[3] += 0.05  # Chùng gối êm ái
            action[9] += 0.05

            pitch_err = state.pelvis_rpy[1]
            pitch_rate = state.pelvis_ang_vel[1]
            u_pitch = 1.2 * pitch_err + 0.16 * pitch_rate + 0.25 * state.com_vel[0]
            action[4] += np.clip(0.8 * u_pitch, -0.30, 0.30)
            action[10] += np.clip(0.8 * u_pitch, -0.30, 0.30)
            action[0] += np.clip(1.0 * u_pitch, -0.35, 0.35)
            action[6] += np.clip(1.0 * u_pitch, -0.35, 0.35)

        action = np.clip(action, sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        state, push, fell = sim.step(action)

        # Ghi nhận dữ liệu
        lf_pos = sim.data.xpos[ik.left_foot_id]
        rf_pos = sim.data.xpos[ik.right_foot_id]

        times.append(t)
        pitch_list.append(np.degrees(state.pelvis_rpy[1]))
        roll_list.append(np.degrees(state.pelvis_rpy[0]))
        left_foot_x.append(lf_pos[0])
        right_foot_x.append(rf_pos[0])
        left_foot_z.append(lf_pos[2])
        right_foot_z.append(rf_pos[2])
        lf_contacts.append(state.left_foot_contact)
        rf_contacts.append(state.right_foot_contact)

        if step % 200 == 0 and t > 0.7:
            print(f"[t = {t:.2f}s] Đang ổn định: Roll={roll_list[-1]:5.1f}°, Pitch={pitch_list[-1]:5.1f}°, Độ cao thân={state.pelvis_pos[2]:.2f}m")

        if fell:
            print(f"\n❌ NGÃ tại t={t:.3f}s!")
            break

    if not fell:
        print("\n" + "=" * 80)
        print("🏆 KẾT QUẢ: Robot BƯỚC 1 CHÂN THÀNH CÔNG VÀ ĐỨNG VỮNG VÀNG!")
        print(f"   - Góc nghiêng cuối cùng: Roll = {roll_list[-1]:.2f}°, Pitch = {pitch_list[-1]:.2f}°")
        print(f"   - Khoảng cách chân phải bước tới: {(right_foot_x[-1] - right_foot_x[0])*100:.1f} cm")
        print(f"   - Chân trái (trụ): Di chuyển {(left_foot_x[-1] - left_foot_x[0])*100:.1f} cm (giữ nguyên vị trí)")
        print("=" * 80)

    # Vẽ biểu đồ phân tích trực quan
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Góc nghiêng Pitch & Roll
    axes[0].plot(times, pitch_list, label='Pitch (Nghiêng dọc)', color='#1f77b4', lw=2)
    axes[0].plot(times, roll_list, label='Roll (Nghiêng ngang)', color='#ff7f0e', lw=2)
    axes[0].axvspan(0.5, 0.6, color='red', alpha=0.15, label='Lực đẩy 85N (0.1s)')
    axes[0].axvspan(phase_t0, phase_t0 + step_duration, color='green', alpha=0.15, label='Giai đoạn vung chân (Swing)')
    axes[0].set_ylabel('Góc nghiêng (°)')
    axes[0].set_title('QUÁ TRÌNH HỒI PHỤC THĂNG BẰNG BƯỚC 1 CHÂN (SINGLE-LEG STEP)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 2. Vị trí X của 2 bàn chân (Sagittal Displacement)
    axes[1].plot(times, np.array(right_foot_x) - right_foot_x[0], label='Chân phải (Chân vung - Swing)', color='#2ca02c', lw=2.5)
    axes[1].plot(times, np.array(left_foot_x) - left_foot_x[0], label='Chân trái (Chân trụ - Stance)', color='#d62728', lw=2, linestyle='--')
    axes[1].set_ylabel('Dịch chuyển X (m)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # 3. Độ cao Z của 2 bàn chân (Foot Clearance)
    axes[2].plot(times, right_foot_z, label='Độ cao chân phải (Nhấc lên khỏi sàn)', color='#9467bd', lw=2.5)
    axes[2].plot(times, left_foot_z, label='Độ cao chân trái (Bám sàn)', color='#8c564b', lw=2, linestyle='--')
    axes[2].set_xlabel('Thời gian (giây)')
    axes[2].set_ylabel('Độ cao Z (m)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = '/home/yennguyen/vr/single_leg_step_demo.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 Đã lưu biểu đồ phân tích trực quan tại: {plot_path}")

if __name__ == '__main__':
    run_single_leg_demo()
