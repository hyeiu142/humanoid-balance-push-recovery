# Humanoid Balance & Push Recovery

Dự án nghiên cứu và phát triển giải thuật giữ thăng bằng và phục hồi khi bị tác động ngoại lực (Push Recovery) cho robot Humanoid trên nền tảng mô phỏng **MuJoCo 3** với mô hình **Unitree G1** (29-DOF, 12-DOF chân với 6-DOF mỗi chân gồm Ankle Pitch & Ankle Roll).

Lộ trình tiến hóa của dự án:
```text
V1: Virtual Model Control (VMC) / Task-Space PID (Baseline)  [HOÀN THÀNH]
      ↓
V2: Model Predictive Control (MPC / LIPM / Stepping Recovery) [HOÀN THÀNH]
      ↓
V3: Reinforcement Learning (PPO / SAC Policy)
      ↓
V4: Hybrid (Learned Policy + QP/MPC Low-Level Control)
```

---

## 1. Cấu trúc thư mục

```text
├── models/
│   └── unitree_g1/                # Mô hình MJCF Unitree G1 từ MuJoCo Menagerie
│       ├── g1.xml
│       ├── scene.xml
│       └── assets/
├── sim/
│   ├── __init__.py
│   ├── simulation_base.py         # MuJoCo simulation environment, reset, step
│   ├── sensors.py                 # State estimator (CoM, CoP, IMU Roll/Pitch, Foot contacts)
│   └── disturbance.py             # Quản lý tiêm xung lực đẩy (impulse push)
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py        # Interface chuẩn cho V1, V2, V3, V4
│   ├── v1_pid/
│   │   ├── __init__.py
│   │   ├── vmc_pid_controller.py  # Bộ điều khiển Virtual Model Control PID
│   │   └── balance_strategies.py  # Điều phối Ankle Strategy & Hip Strategy
│   └── v2_mpc/
│       ├── __init__.py
│       ├── lipm_model.py          # Mô hình Linear Inverted Pendulum & Capture Point
│       ├── qp_mpc_solver.py       # Bộ giải QP tối ưu hóa ZMP/CoM (OSQP <0.2ms)
│       ├── leg_kinematics.py      # DLS Inverse Kinematics 6-DOF & Foot Sole Leveling
│       └── mpc_controller.py      # Bộ điều khiển tích hợp Hierarchical MPC & Stepping
├── benchmark/
│   ├── __init__.py
│   ├── metrics.py                 # Đo settling time, max impulse, CoM/CoP excursion
│   ├── run_push_benchmark.py      # Đánh giá đối đầu 3 bên: Passive vs V1 PID vs V2 MPC
│   ├── v1_benchmark_results.png   # Biểu đồ kết quả benchmark V1
│   └── v2_benchmark_results.png   # Biểu đồ 4 bảng so sánh V1 vs V2 xuất bản
├── scripts/
│   └── run_interactive.py         # Ứng dụng mô phỏng 3D tương tác 60 FPS với MuJoCo Viewer
└── tests/
    ├── test_v1_balance.py         # 5 bài unit test cho V1 (100% Passed)
    └── test_v2_mpc.py             # 6 bài test cho V2 MPC, QP, IK, và Stepping (100% Passed)
```

---

## 2. Hướng dẫn cài đặt & Chạy

### 2.1. Kích hoạt môi trường ảo
```bash
source .venv/bin/activate
```

### 2.2. Chạy ứng dụng tương tác 3D (Interactive Viewer)
Mở cửa sổ 3D của MuJoCo để quan sát robot đứng và tự tay tác động lực đẩy theo thời gian thực:
```bash
.venv/bin/python scripts/run_interactive.py
```

**Phím điều khiển:**
* **Phím M hoặc C**: Chuyển đổi vòng lặp giữa 3 bộ điều khiển:
  * `V2: MPC Stepping` (Mặc định - Khả năng chịu lực cực hạn **220N+**)
  * `Passive Baseline` (Tắt controller để xem robot ngã)
  * `V1: VMC PID` (Controller phản hồi V1)
* **Phím MŨI TÊN (Lên / Xuống / Trái / Phải)**: Tác động lực đẩy thủ công.
* **Phím số 1..5**: Chọn độ mạnh của cú đẩy:
  * `1`: Đẩy nhẹ (30 N)
  * `2`: Đẩy vừa (70 N)
  * `3`: Đẩy mạnh (120 N)
  * `4`: Đẩy cực hạn V1 (150 N)
  * `5`: Đẩy siêu mạnh V2 Stepping (220 N)
* **Phím D**: Bật / Tắt chế độ Auto-Demo (tự động đẩy theo kịch bản từ 30N đến 220N).
* **Phím R**: Reset robot về tư thế đứng thẳng ban đầu.
* **Spacebar**: Lặp lại cú đẩy vừa chọn.

### 2.3. Chạy Suite Benchmark đối đầu 3 bên (Head-to-Head)
Quét toàn bộ các mức lực theo 3 hướng (Forward, Backward, Lateral) và tự động xuất bảng so sánh 3 bên cùng đồ thị 4 bảng:
```bash
.venv/bin/python benchmark/run_push_benchmark.py
```
Kết quả biểu đồ sẽ được lưu tại: `benchmark/v2_benchmark_results.png`.

### 2.4. Chạy Automated Tests (11/11 Passed)
```bash
.venv/bin/python -m unittest discover -s tests -v
```

---

## 3. Kết quả Benchmark So Sánh Đối Đầu (Passive vs V1 PID vs V2 MPC)

### 3.1. Bảng số liệu tổng hợp

| Hướng tác động | Lực đẩy ($0.1\text{s}$) | Xung lực ($J$) | Passive Baseline | V1: VMC PID | V2: MPC Stepping | V2 Độ nghiêng max | V2 Thời gian ổn định |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forward (+X)** | 30 N | 3.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.16°** | 0.00s |
| **Forward (+X)** | 70 N | 7.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.40°** | 0.00s |
| **Forward (+X)** | 120 N | 12.0 N·s | FELL (2.60s) | SURVIVED | **SURVIVED** | **0.80°** | 0.00s |
| **Forward (+X)** | 150 N | 15.0 N·s | FELL (2.75s) | SURVIVED (28.4°) | **SURVIVED** | **1.64°** | 2.40s |
| **Forward (+X)** | 180 N | 18.0 N·s | FELL (1.74s) | FELL (1.76s) | **SURVIVED** | **1.68°** | 2.40s |
| **Forward (+X)** | 220 N | 22.0 N·s | FELL (1.43s) | FELL (1.50s) | **SURVIVED (Step!)** | **3.37°** | 2.40s |
| **Forward (+X)** | 260 N | 26.0 N·s | FELL (1.29s) | FELL (1.36s) | FELL (2.42s) | 41.06° | N/A |
| **Backward (-X)** | 30 N | 3.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.20°** | 0.00s |
| **Backward (-X)** | 60 N | 6.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.46°** | 0.00s |
| **Backward (-X)** | 90 N | 9.0 N·s | FELL (1.83s) | FELL (2.03s) | FELL (1.96s) | 37.71° | N/A |
| **Lateral (+Y)** | 40 N | 4.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.06°** | 0.00s |
| **Lateral (+Y)** | 80 N | 8.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.20°** | 0.00s |
| **Lateral (+Y)** | 120 N | 12.0 N·s | SURVIVED | SURVIVED | **SURVIVED** | **0.29°** | 0.00s |

### 3.2. So sánh giới hạn chịu đựng cực đại (Maximum Tolerated Impulse)

* **Hướng tới trước (Forward +X)**:
  * **Passive**: $7.0\text{ N}\cdot\text{s}$ (70 N)
  * **V1 PID**: $15.0\text{ N}\cdot\text{s}$ (150 N, tăng +114% so với Passive)
  * **V2 MPC**: **$22.0\text{ N}\cdot\text{s}$ (220 N, tăng +47% so với V1, tăng +214% so với Passive!)**
* **Góc nghiêng thân (Torso Tilt Suppression)**:
  * Ở lực 150 N: V1 PID bị nghiêng tới **28.4°**, trong khi V2 MPC duy trì thân thẳng tắp chỉ nghiêng **1.64°** (giảm rung lắc hơn **17 lần**!).
  * Ở lực 220 N: Cả Passive và V1 PID đều ngã lộn nhào ở giây 1.43s - 1.50s, trong khi V2 MPC kích hoạt bước chân đón đầu Capture Point và ổn định thân với góc nghiêng chỉ **3.37°**!

---

## 4. Những cải tiến kỹ thuật cốt lõi ở V2

1. **Bộ giải QP OSQP siêu nhanh (<0.13 ms)**: Tối ưu hóa quỹ đạo ZMP trong chân trời dự đoán 16 bước thời gian thực trên CPU với cơ chế warm-start.
2. **Instantaneous Capture Point (ICP / DCM)**: Dự phóng vị trí điểm tiếp đất tương lai $\xi(T_{step}) = \xi_0 e^{\omega_0 T_{step}}$ để đặt chân đón đầu năng lượng động năng.
3. **Khóa phẳng đế bàn chân (Foot Sole Leveling Constraint)**: Ràng buộc hình học $q_{ankle\_pitch} = -(q_{hip\_pitch} + q_{knee} + \theta_{pelvis})$ giữ mặt đế bàn chân luôn song song với sàn nhà khi co gập chân, loại bỏ hoàn toàn hiện tượng tiếp đất bằng mũi ngón chân gây lật gót.
4. **DLS Inverse Kinematics 6-DOF chính xác cao**: Sai số vị trí dưới $0.065\text{ mm}$, đảm bảo điều khiển bàn chân mượt mà.
