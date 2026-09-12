# Walkthrough: Humanoid Balance & Push Recovery (V1 - Baseline PID)

Đã hoàn thành toàn bộ việc xây dựng môi trường mô phỏng vật lý, mô hình robot Humanoid **Unitree G1**, bộ điều khiển **V1: Virtual Model Control (VMC) / Task-Space PID** với cơ chế **Ankle Strategy & Hip Strategy**, ứng dụng tương tác **3D Viewer** và hệ thống **Automated Benchmark Suite**.

---

## 1. Các thành phần đã triển khai

| Thành phần | Đường dẫn | Mô tả chức năng |
| :--- | :--- | :--- |
| **Simulation Base** | [simulation_base.py](file:///home/yennguyen/vr/sim/simulation_base.py) | Quản lý vòng lặp MuJoCo, nạp keyframe đứng, đồng bộ hóa bước mô phỏng và tiêm ngoại lực. |
| **State Estimator & Sensors** | [sensors.py](file:///home/yennguyen/vr/sim/sensors.py) | Tính toán chính xác trọng tâm CoM, áp lực tiếp xúc, điểm đặt lực CoP, góc nghiêng IMU và đa giác hỗ trợ (Support Polygon). |
| **Disturbance Manager** | [disturbance.py](file:///home/yennguyen/vr/sim/disturbance.py) | Tiêm xung lực đẩy ($F_x, F_y, F_z$) theo thời gian thực và theo lịch trình. |
| **Controller Interface** | [base_controller.py](file:///home/yennguyen/vr/controllers/base_controller.py) | Lớp trừu tượng chuẩn hóa đầu vào/đầu ra cho cả 4 phiên bản (V1, V2, V3, V4). |
| **V1 VMC PID Controller** | [vmc_pid_controller.py](file:///home/yennguyen/vr/controllers/v1_pid/vmc_pid_controller.py) | Ánh xạ mô-men ảo từ góc nghiêng thân và vị trí CoM xuống 12 khớp chân (cổ chân và hông). |
| **Strategy Coordinator** | [balance_strategies.py](file:///home/yennguyen/vr/controllers/v1_pid/balance_strategies.py) | Tự động chuyển đổi mượt mà giữa Ankle Strategy (đẩy nhẹ) và Hip Strategy (đẩy vừa). |
| **Benchmark Suite** | [run_push_benchmark.py](file:///home/yennguyen/vr/benchmark/run_push_benchmark.py) | Quét tự động dải lực, đo thời gian ổn định ($t_{settle}$), xung lực cực đại ($J_{max}$) và vẽ đồ thị 4 panels. |
| **Interactive 3D App** | [scripts/run_interactive.py](file:///home/yennguyen/vr/scripts/run_interactive.py) | Cửa sổ 3D MuJoCo Viewer cho phép bấm phím điều khiển đẩy robot theo thời gian thực. |
| **Automated Tests** | [test_v1_balance.py](file:///home/yennguyen/vr/tests/test_v1_balance.py) | Bộ 5 bài test tự động kiểm tra cảm biến, tư thế đứng tĩnh, phản xạ đẩy nhẹ/vừa và giới hạn ngã. |

---

## 2. Kết quả kiểm chứng và Benchmark

### 2.1. Đồ thị phân tích Benchmark V1

![Biểu đồ kết quả Benchmark so sánh Passive Baseline và V1 VMC PID](/home/yennguyen/.gemini/antigravity-ide/brain/aaa15f44-c1a2-45f0-b9c1-e9829167c937/v1_benchmark_results.png)

### 2.2. Bảng tổng hợp số liệu kiểm thử thực tế

```text
===============================================================================================
Direction       | Force (N) | Passive Status   | V1 PID Status    | V1 Settle (s) | V1 Max Tilt
-----------------------------------------------------------------------------------------------
Forward (+X)    |      30.0 | SURVIVED         | SURVIVED         | 0.00s         |     0.51°
Forward (+X)    |      70.0 | SURVIVED         | SURVIVED         | 0.14s         |     1.35°
Forward (+X)    |     100.0 | SURVIVED         | SURVIVED         | 0.29s         |     2.10°
Forward (+X)    |     120.0 | FELL (2.60s)     | SURVIVED         | 0.43s         |     2.66°
Forward (+X)    |     150.0 | FELL (2.75s)     | SURVIVED         | 2.40s         |    28.43°
Forward (+X)    |     180.0 | FELL (1.74s)     | FELL (1.77s)     | N/A           |    40.82°
Backward (-X)   |      30.0 | SURVIVED         | SURVIVED         | 0.00s         |     0.61°
Backward (-X)   |      60.0 | SURVIVED         | SURVIVED         | 0.44s         |     1.74°
Backward (-X)   |      90.0 | FELL (1.83s)     | FELL (2.03s)     | N/A           |    30.55°
Lateral (+Y)    |      40.0 | SURVIVED         | SURVIVED         | 0.00s         |     0.06°
Lateral (+Y)    |      80.0 | SURVIVED         | SURVIVED         | 0.00s         |     0.18°
Lateral (+Y)    |     120.0 | SURVIVED         | SURVIVED         | 0.00s         |     0.28°
===============================================================================================
```

### 2.3. Kết quả Automated Tests
Đã chạy `python -m unittest tests/test_v1_balance.py -v`:
- `test_sensors_and_cop`: **PASS** (Lực pháp tuyến khớp 327N trọng lực, CoP nằm chính giữa đa giác hỗ trợ).
- `test_static_standing_stability`: **PASS** (Đứng tĩnh 3s ổn định, độ nghiêng $< 0.04^\circ$).
- `test_mild_push_ankle_recovery`: **PASS** (Đẩy 30N hồi phục hoàn toàn bằng Ankle Strategy).
- `test_medium_push_hip_recovery`: **PASS** (Đẩy 120N: robot thụ động ngã, V1 hồi phục sau 0.43s bằng Hip Strategy).
- `test_physical_tipping_limit`: **PASS** (Đẩy 200N vượt quá đa giác hỗ trợ, phát hiện ngã chính xác).

---

## 3. Cách chạy thử nghiệm

### 1. Trải nghiệm tương tác trực tiếp trên giao diện 3D (Interactive Viewer)
Mở terminal và gõ:
```bash
.venv/bin/python scripts/run_interactive.py
```
- Sử dụng các phím **Mũi tên** (Up/Down/Left/Right) để đẩy robot theo các hướng.
- Bấm phím **1, 2, 3, 4** để đổi lực đẩy từ 30N đến 150N.
- Bấm phím **C** để bật/tắt controller xem sự khác biệt giữa có controller và không có controller.
- Bấm phím **R** để reset robot đứng thẳng.

### 2. Chạy lại benchmark tự động
```bash
.venv/bin/python benchmark/run_push_benchmark.py
```
Biểu đồ 4-panel sẽ được cập nhật tại [v1_benchmark_results.png](file:///home/yennguyen/vr/benchmark/v1_benchmark_results.png).
