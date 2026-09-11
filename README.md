# Humanoid Balance & Push Recovery

Dự án nghiên cứu và phát triển giải thuật giữ thăng bằng và phục hồi khi bị tác động ngoại lực (Push Recovery) cho robot Humanoid trên nền tảng mô phỏng **MuJoCo 3** với mô hình **Unitree G1** (29-DOF, 12-DOF chân với 6-DOF mỗi chân gồm Ankle Pitch & Ankle Roll).

Lộ trình tiến hóa của dự án:
```text
V1: Virtual Model Control (VMC) / Task-Space PID (Baseline)  [HOÀN THÀNH]
      ↓
V2: Model Predictive Control (MPC / LIPM / ZMP Preview)
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
│   └── v1_pid/
│       ├── __init__.py
│       ├── vmc_pid_controller.py  # Bộ điều khiển Virtual Model Control PID
│       └── balance_strategies.py  # Điều phối Ankle Strategy & Hip Strategy
├── benchmark/
│   ├── __init__.py
│   ├── metrics.py                 # Đo settling time, max impulse, CoM/CoP excursion
│   ├── run_push_benchmark.py      # Quét tự động dải lực và xuất đồ thị so sánh
│   └── v1_benchmark_results.png   # Biểu đồ kết quả benchmark V1
├── scripts/
│   └── run_interactive.py         # Ứng dụng mô phỏng 3D tương tác với MuJoCo Viewer
└── tests/
    └── test_v1_balance.py         # Bộ unit & integration test tự động (5/5 passed)
```

---

## 2. Hướng dẫn cài đặt & Chạy

### 2.1. Kích hoạt môi trường ảo
```bash
source .venv/bin/activate
```
*(Các thư viện `mujoco`, `numpy`, `scipy`, `matplotlib` đã được cài đặt sẵn trong `.venv`)*

### 2.2. Chạy ứng dụng tương tác 3D (Interactive Viewer)
Mở cửa sổ 3D của MuJoCo để quan sát robot đứng và tự tay tác động lực đẩy theo thời gian thực:
```bash
.venv/bin/python scripts/run_interactive.py
```

**Phím điều khiển:**
* **Phím mũi tên (Up / Down / Left / Right)**: Đẩy robot tới trước (+X), lùi sau (-X), sang trái (+Y), sang phải (-Y).
* **Phím số 1 / 2 / 3 / 4**: Chọn mức lực đẩy:
  * `1`: Đẩy nhẹ (30 N)
  * `2`: Đẩy vừa (70 N)
  * `3`: Đẩy mạnh (120 N)
  * `4`: Đẩy cực hạn (150 N)
* **Spacebar**: Lặp lại cú đẩy vừa chọn.
* **Phím C**: Chuyển đổi qua lại giữa `V1: VMC PID` và `Passive Baseline` (để thấy rõ sự khác biệt khi có và không có controller).
* **Phím R**: Đặt lại robot về tư thế đứng thẳng chuẩn.

### 2.3. Chạy Suite Benchmark tự động
Quét toàn bộ các mức lực theo 3 hướng (Forward, Backward, Lateral) và tự động xuất bảng so sánh và đồ thị:
```bash
.venv/bin/python benchmark/run_push_benchmark.py
```
Kết quả biểu đồ sẽ được lưu tại: `benchmark/v1_benchmark_results.png`.

### 2.4. Chạy Automated Tests
```bash
.venv/bin/python -m unittest tests/test_v1_balance.py -v
```

---

## 3. Kết quả Benchmark V1 (Baseline PID)

### 3.1. Bảng số liệu so sánh: Passive Baseline vs V1 VMC PID

| Hướng tác động | Lực đẩy ($0.1\text{s}$) | Xung lực ($J$) | Passive Baseline | V1 VMC PID | V1 Thời gian ổn định ($t_{settle}$) | V1 Độ nghiêng lớn nhất |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forward (+X)** | 30 N | 3.0 N·s | SURVIVED | **SURVIVED** | 0.00s | 0.51° |
| **Forward (+X)** | 70 N | 7.0 N·s | SURVIVED | **SURVIVED** | 0.14s | 1.35° |
| **Forward (+X)** | 100 N | 10.0 N·s | SURVIVED | **SURVIVED** | 0.29s | 2.10° |
| **Forward (+X)** | 120 N | 12.0 N·s | **FELL (ngã tại 2.60s)** | **SURVIVED** | 0.43s | 2.66° |
| **Forward (+X)** | 150 N | 15.0 N·s | **FELL (ngã tại 2.75s)** | **SURVIVED** (Kích hoạt Hip) | 2.40s | 28.43° |
| **Forward (+X)** | 180 N | 18.0 N·s | FELL (1.74s) | **FELL (1.77s)** | N/A | 40.82° |
| **Backward (-X)** | 30 N | 3.0 N·s | SURVIVED | **SURVIVED** | 0.00s | 0.61° |
| **Backward (-X)** | 60 N | 6.0 N·s | SURVIVED | **SURVIVED** | 0.44s | 1.74° |
| **Backward (-X)** | 90 N | 9.0 N·s | **FELL (1.83s)** | **FELL (2.03s)** | N/A | 30.55° |
| **Lateral (+Y)** | 40 N | 4.0 N·s | SURVIVED | **SURVIVED** | 0.00s | 0.06° |
| **Lateral (+Y)** | 80 N | 8.0 N·s | SURVIVED | **SURVIVED** | 0.00s | 0.18° |
| **Lateral (+Y)** | 120 N | 12.0 N·s | SURVIVED | **SURVIVED** | 0.00s | 0.28° |

### 3.2. Phân tích kết luận vật lý cốt lõi:
1. **Khả năng chịu xung lực tới trước (+X)**:
   - Bộ điều khiển **V1 VMC PID tăng 50% khả năng chịu xung lực** (từ $10.0\text{ N}\cdot\text{s}$ lên $15.0\text{ N}\cdot\text{s}$).
   - Ở mức 120N, robot thụ động dao động mất kiểm soát và ngã sấp ở giây 2.60, trong khi V1 dập tắt dao động chỉ sau **0.43 giây**.
   - Ở mức 150N, **Hip Strategy** được kích hoạt: robot gập hông về phía trước để triệt tiêu mô-men quán tính lật, giữ cho CoP không vượt ra khỏi mũi bàn chân.
2. **Giới hạn vật lý của giữ thăng bằng tại chỗ**:
   - Khi lực đẩy vượt quá $150\text{N}$ ($>15\text{N}\cdot\text{s}$), CoP bị đẩy chạm tới mép đầu ngón chân ($x = +0.12\text{m}$). Tại đây, robot bắt buộc phải nhấc chân bước một bước (**Stepping Recovery / Capture Point**) mới không bị ngã.
   - Khi bị đẩy lui sau ($-X$), do khoảng cách từ cổ chân tới gót chỉ là $5\text{cm}$ (ngắn hơn nhiều so với $12\text{cm}$ từ cổ chân tới mũi), giới hạn lật gót xảy ra ở mức $6.0\text{ N}\cdot\text{s}$.
3. **Cơ sở cho giai đoạn tiếp theo (V2 MPC & V3 RL)**:
   - Các chỉ số trên cung cấp baseline định lượng chuẩn xác. Ở V2 (MPC), ta sẽ xây dựng mô hình con lắc ngược tuyến tính (LIPM) kết hợp tối ưu hóa quỹ đạo CoP/ZMP để mở rộng khả năng thăng bằng và thực hiện bước chân phục hồi.
