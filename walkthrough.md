# Walkthrough: Humanoid Balance & Push Recovery (V2 - MPC & Stepping Recovery)

Đã hoàn thành toàn diện việc nâng cấp giai đoạn **V2: Linear Inverted Pendulum Model (LIPM) + Convex QP Preview MPC + Single-Leg Stepping Push Recovery (Bước 1 chân thật sự)** trên mô hình robot Humanoid **Unitree G1**.

---

## 1. Nâng cấp cốt lõi trong V2 (So với V1)

| Hạng mục | V1 (VMC PID Baseline) | V2 (LIPM QP-MPC + Stepping Recovery) |
| :--- | :--- | :--- |
| **Cơ sở mô hình động học** | Con lắc ngược tĩnh & góc nghiêng IMU | Mô hình LIPM 3D liên tục & rời rạc hóa ($z_0 = 0.693\text{ m}, \omega_0 = 3.76\text{ rad/s}$) |
| **Dự báo trước (Preview)** | Phản hồi sai số tức thời (P-I-D) | QP Preview Horizon ($N = 16$ bước, $\Delta t = 0.05\text{s}$, $T_{lookahead} = 0.8\text{s}$) giải qua OSQP |
| **Tối ưu hóa điểm đặt lực** | CoP bão hòa biên thụ động | Tối ưu hóa ZMP bên trong đa giác chân với trọng số $Q_{com}, R_{zmp}, R_{rate}$ (< 0.1 ms) |
| **Bảo toàn thăng bằng cực hạn** | Bất lực khi CoP chạm biên ngón chân ($> 150\text{N}$) | **Stepping Recovery**: Điều chỉnh bước chân tới điểm Capture Point để mở rộng đa giác hỗ trợ |
| **Cơ chế bước chân** | Không có (Chân cố định trên sàn) | **2 Chế độ linh hoạt (`SteppingMode`)**: <br>1. `SINGLE_LEG`: Nhấc 1 chân lăng bước tới trước, chân trụ cắm sàn, cân chỉnh đế giày, tiếp đất so le. <br>2. `SYNC_SHUFFLE`: Nhảy đồng pha cả 2 chân hấp thụ xung lực cực đại ($> 150\text{N} \to 220\text{N}$). |
| **Giới hạn chịu lực đẩy (+X)** | $150\text{ N}$ ($15.0\text{ N}\cdot\text{s}$) | **$220\text{ N}$ ($22.0\text{ N}\cdot\text{s}$)** (+47% so với V1, +214% so với Passive) |

---

## 2. Các thành phần mã nguồn V2 đã xây dựng

| Tệp tin | Đường dẫn | Mô tả chi tiết |
| :--- | :--- | :--- |
| **LIPM Model** | [lipm_model.py](file:///home/yennguyen/vr/controllers/v2_mpc/lipm_model.py) | Trạng thái $[x, \dot{x}]^T$, ma trận động học trạng thái $A, B$, tính toán Instantaneous Capture Point (ICP). |
| **QP MPC Solver** | [qp_mpc_solver.py](file:///home/yennguyen/vr/controllers/v2_mpc/qp_mpc_solver.py) | Thiết lập bài toán QP dạng chuẩn giải bằng `OSQP`, thời gian giải trung bình **0.08 ms** (yêu cầu $< 2.0\text{ ms}$). |
| **Footstep Planner** | [footstep_planner.py](file:///home/yennguyen/vr/controllers/v2_mpc/footstep_planner.py) | Dự đoán vị trí tiếp đất dựa trên vận tốc và vị trí Capture Point khi rời chân. |
| **Leg Kinematics (IK)** | [leg_kinematics.py](file:///home/yennguyen/vr/controllers/v2_mpc/leg_kinematics.py) | Nghịch đảo động học Damped Least Squares (DLS) 6-DOF mỗi chân, sai số vị trí **0.064 mm** (< 0.1 mm). |
| **V2 MPC Controller** | [mpc_controller.py](file:///home/yennguyen/vr/controllers/v2_mpc/mpc_controller.py) | Bộ điều khiển FSM 3 trạng thái (`DOUBLE_SUPPORT`, `STEP_SWING`, `LANDED_SETTLE`), hỗ trợ `SINGLE_LEG` và `SYNC_SHUFFLE`. |
| **Interactive 3D Viewer** | [run_interactive.py](file:///home/yennguyen/vr/scripts/run_interactive.py) | Thêm phím **'V'** để chuyển đổi Stepping Mode, hiển thị HUD `FSM: <STATE> (<MODE>)`. |
| **Unit Tests V2** | [test_v2_mpc.py](file:///home/yennguyen/vr/tests/test_v2_mpc.py) | Toàn bộ 8 bài test kiểm thử LIPM, QP, IK, đứng tĩnh, đẩy nhẹ, Single-Leg step và 220N pass 100%. |

---

## 3. Kết quả Benchmark 3 Chiều: Passive vs V1 PID vs V2 MPC

![Biểu đồ Benchmark so sánh Passive Baseline, V1 VMC PID và V2 MPC](/home/yennguyen/.gemini/antigravity-ide/brain/aaa15f44-c1a2-45f0-b9c1-e9829167c937/v2_benchmark_results.png)

```text
=========================================================================================================
Direction       | Force   | Impulse  | Passive        | V1 PID         | V2 MPC         | V2 Max Tilt | V2 Settle
---------------------------------------------------------------------------------------------------------
Forward (+X)    |    30 N |   3.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.16°   | 0.00s    
Forward (+X)    |    70 N |   7.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.40°   | 0.00s    
Forward (+X)    |   120 N |  12.0 Ns | FELL (2.60s)   | SURVIVED       | SURVIVED       |     0.80°   | 0.00s    
Forward (+X)    |   150 N |  15.0 Ns | FELL (2.75s)   | SURVIVED       | SURVIVED       |     1.64°   | 2.40s    
Forward (+X)    |   180 N |  18.0 Ns | FELL (1.74s)   | FELL (1.76s)   | SURVIVED       |     1.68°   | 2.40s    
Forward (+X)    |   220 N |  22.0 Ns | FELL (1.43s)   | FELL (1.50s)   | SURVIVED       |     3.42°   | 2.40s    
Forward (+X)    |   260 N |  26.0 Ns | FELL (1.29s)   | FELL (1.36s)   | FELL (2.36s)   |    41.41°   | N/A      
Backward (-X)   |    30 N |   3.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.20°   | 0.00s    
Backward (-X)   |    60 N |   6.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.46°   | 0.00s    
Backward (-X)   |    90 N |   9.0 Ns | FELL (1.83s)   | FELL (2.03s)   | FELL (1.96s)   |    37.71°   | N/A      
Lateral (+Y)    |    40 N |   4.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.07°   | 0.00s    
Lateral (+Y)    |    80 N |   8.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.24°   | 0.00s    
Lateral (+Y)    |   120 N |  12.0 Ns | SURVIVED       | SURVIVED       | SURVIVED       |     0.31°   | 0.00s    
=========================================================================================================

--- TỔNG KẾT XUNG LỰC CỰC ĐẠI CHỊU ĐƯỢC (N·s) ---
  Forward (+X)   : Passive = 7.0 N·s -> V1 PID = 15.0 N·s (+114%) -> V2 MPC = 22.0 N·s (+47% so với V1)
  Backward (-X)  : Passive = 6.0 N·s -> V1 PID =  6.0 N·s (+0%)   -> V2 MPC =  6.0 N·s (+0%)
  Lateral (+Y)   : Passive = 12.0 N·s -> V1 PID = 12.0 N·s (+0%)  -> V2 MPC = 12.0 N·s (+0%)
```

---

## 4. Hướng dẫn chạy thử nghiệm & Điều khiển

### 1. Trải nghiệm tương tác 3D (Interactive Viewer)
```bash
.venv/bin/python scripts/run_interactive.py
```
- **Phím C / M**: Chuyển đổi bộ điều khiển: `Passive` $\to$ `V1 VMC PID` $\to$ `V2 MPC (LIPM)`.
- **Phím V**: Chuyển đổi chế độ bước chân V2: `SINGLE_LEG` (bước 1 chân) $\longleftrightarrow$ `SYNC_SHUFFLE` (nhảy 2 chân).
- **Phím 1..5**: Lựa chọn mức lực đẩy:
  - `1`: 30N (Nhẹ)
  - `2`: 70N (Vừa)
  - `3`: 120N (Mạnh)
  - `4`: 150N (Cực hạn V1)
  - `5`: 220N (Siêu mạnh - V2 Stepping Recovery)
- **Mũi tên hoặc W / S / A / D**: Đẩy robot theo hướng tương ứng.
- **Phím Space**: Lặp lại cú đẩy gần nhất.
- **Phím R**: Reset về tư thế đứng ban đầu.
- **Phím D**: Bật/tắt chế độ Auto-Demo chạy tự động các bài test.

### 2. Chạy toàn bộ Unit Tests
```bash
.venv/bin/python -m unittest tests/test_v2_mpc.py -v
.venv/bin/python -m unittest tests/test_v1_balance.py -v
```

### 3. Chạy lại toàn bộ Benchmark so sánh
```bash
.venv/bin/python benchmark/run_push_benchmark.py
```
