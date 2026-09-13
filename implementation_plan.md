# Implementation Plan: Humanoid Balance & Push Recovery (V2 - Model Predictive Control & Stepping Recovery)

Tài liệu thiết kế chi tiết triển khai giai đoạn **V2: Model Predictive Control (MPC) & Stepping Push Recovery** cho robot Humanoid **Unitree G1**, kế thừa và phát triển trực tiếp trên nền tảng mô phỏng MuJoCo đã hoàn thiện ở giai đoạn V1. Toàn bộ các quyết định kỹ thuật và kiến trúc đã được đồng thuận và kiểm chứng thông qua quy trình stress-test (`/grill-me`).

---

## 1. Mục tiêu và Sự vượt trội của V2 so với V1

Ở giai đoạn V1, bộ điều khiển PID hoạt động theo nguyên lý **phản ứng thụ động (Reactive Control)**: khi bị đẩy nghiêng thì mới đo sai số để bù góc. Giới hạn tối đa của việc giữ thăng bằng tại chỗ là **$15.0\text{ N}\cdot\text{s}$ (150N trong 0.1s)**. Khi lực đẩy vượt quá 150N, năng lượng xung va chạm vượt quá rào cản thế năng của bàn chân 12cm ($E_{push} > 4.86\text{ J} > E_{barrier} \approx 3.27\text{ J}$), và robot bắt buộc phải ngã nếu đứng yên một chỗ.

**Giai đoạn V2 giải quyết triệt để vấn đề này nhờ hệ thống phân tầng kép (Hierarchical Balance Framework):**
1. **MPC Dự đoán tương lai (Predictive In-Place Optimization - Lực $\le 150\text{N}$)**:
   - Sử dụng mô hình con lắc ngược tuyến tính (**LIPM**) và bộ giải **Quadratic Programming (OSQP)** để nhìn trước chân trời thời gian $N = 16$ bước ($0.8\text{s}$ tới).
   - Tối ưu hóa quỹ đạo CoM và CoP/ZMP mượt mà, dập tắt dao động nhanh hơn và êm hơn PID ở dải lực $< 150\text{N}$, giữ góc nghiêng thân cực nhỏ ($< 3.7^\circ$).
2. **Bước chân phục hồi (Capture Point Stepping Recovery - Lực $150\text{N} - 350\text{N}$)**:
   - Khi lực đẩy vượt quá ngưỡng đa giác hỗ trợ ($> 150\text{N}$, thử nghiệm tới **$250\text{N} - 300\text{N}+$**), thuật toán **Instantaneous Capture Point (ICP / DCM)** tự động tính toán thời điểm và tọa độ đặt chân đón đầu $p_{target} = \xi_{touchdown}$.
   - Chân lăng bước một bước về phía trước/bên để mở rộng đa giác hỗ trợ (tăng rào cản thế năng chống lật lên gấp $10\times$, đạt $30\text{ J}$), dập tắt hoàn toàn xung lực và cứu robot khỏi cú ngã.

---

## 2. Kiến trúc hệ thống phân tầng 2 cấp (Two-Level Hierarchy)

```text
               ┌────────────────────────────────────────────────────────┐
               │                     MuJoCo Physics                     │
               │             (Unitree G1 MJCF Menagerie)                │
               └──────────────┬──────────────────────────▲──────────────┘
                              │ State (CoM, IMU, Contacts)│ 500 Hz Joint Commands
                              ▼                          │ (12 Leg Actuators)
               ┌─────────────────────────────────────────┴──────────────┐
               │     Low-Level Tracking Controller (500 Hz)             │
               │  - Analytical Inverse Kinematics (IK 6-DOF per leg)    │
               │  - Foot Sole Leveling (Giữ đế bàn chân song song sàn) │
               │  - Stance Anti-Buckling Height & Hip/Torso Damping     │
               │  - Swing Trajectory (Minimum-Jerk Cycloid 3D)          │
               │  - Active Landing Compliance & Ankle Braking           │
               └──────────────────────────▲─────────────────────────────┘
                                          │ Reference Trajectories:
                                          │ - CoM trajectory x_com(t)
                                          │ - ZMP trajectory zmp(t)
                                          │ - Footstep target p_step(t)
               ┌──────────────────────────┴─────────────────────────────┐
               │     High-Level MPC & Step Planner (50 Hz / 20 Hz)      │
               │  - Linear Inverted Pendulum Model (LIPM) Preview       │
               │  - Instantaneous Capture Point (ICP / DCM) Calculator  │
               │  - QP Optimization (OSQP, <1ms warm-start)             │
               │  - FSM: DOUBLE_SUPPORT <-> SWING_STEP <-> WIDE_SETTLE  │
               └────────────────────────────────────────────────────────┘
```

---

## 3. Các module thuật toán cốt lõi

### 3.1. Mô hình con lắc ngược tuyến tính (LIPM Dynamics)
Giả định độ cao trọng tâm CoM giữ không đổi ở $z_0 \approx 0.693\text{m}$. Phương trình vi phân chuyển động của CoM theo phương ngang ($x, y$):
$$\ddot{x} = \omega_0^2 (x - p_{zmp}), \quad \text{với } \omega_0 = \sqrt{\frac{g}{z_0}} \approx 3.76\text{ rad/s}$$

Hệ phương trình trạng thái rời rạc với bước thời gian $\Delta t_{mpc} = 0.05\text{s}$ ($20\text{Hz}$):
$$X_{k+1} = A X_k + B p_{zmp, k}, \quad X_k = \begin{bmatrix} x_k \\ \dot{x}_k \end{bmatrix}$$
$$A = \begin{bmatrix} \cosh(\omega_0 \Delta t) & \frac{1}{\omega_0}\sinh(\omega_0 \Delta t) \\ \omega_0 \sinh(\omega_0 \Delta t) & \cosh(\omega_0 \Delta t) \end{bmatrix}, \quad B = \begin{bmatrix} 1 - \cosh(\omega_0 \Delta t) \\ -\omega_0 \sinh(\omega_0 \Delta t) \end{bmatrix}$$

### 3.2. Thuật toán Capture Point & Kích hoạt bước chân (Single Recovery Step)
Điểm bắt thăng bằng tức thời (Instantaneous Capture Point - ICP / DCM):
$$\xi = x_{com} + \frac{\dot{x}_{com}}{\omega_0}$$

* **Khi $\xi$ nằm trong lòng bàn chân** ($\xi \le x_{toe} + \delta$): Duy trì chế độ **In-place Balance MPC**, tối ưu hóa $p_{zmp}$ và phối hợp góc hông triệt tiêu dao động.
* **Khi $\xi$ vượt ra ngoài mép bàn chân** ($\xi > x_{toe} + \delta$ hoặc $v_{com} > 0.35\text{ m/s}$): Kích hoạt **Stepping Recovery**:
  * **Chọn chân lăng**: Nếu lực lệch bên trái ($+y$), chọn chân trái; nếu lệch bên phải ($-y$), chọn chân phải; nếu đẩy thẳng ($+x$), chọn chân tự do dựa trên pha trước.
  * **Tọa độ tiếp đất dự phóng (Projected Touchdown ICP)**:
    Do Capture Point phân kỳ theo hàm mũ trong thời gian chân lăng đang bay ($T_{step} \approx 0.22\text{s}$):
    $$\xi(T_{step}) = \xi(t_0) e^{\omega_0 T_{step}}$$
    $$p_{step}^{target} = \xi(T_{step}) + \delta_{safety}$$
    Đặt chân đón đầu đúng vị trí này đảm bảo khi chân chạm đất, Capture Point rơi vào đúng tâm của đa giác hỗ trợ mới!

### 3.3. Bài toán tối ưu hóa Quadratic Programming (QP Formulation)
Trong mỗi chu kỳ MPC, giải bài toán QP trong chân trời dự đoán $N = 16$ bước ($0.8\text{s}$):
$$\min_{P_{zmp}} \sum_{k=1}^N \left( Q_x (x_k - x_{ref})^2 + Q_v \dot{x}_k^2 + R (p_{zmp, k} - p_{zmp, k-1})^2 \right)$$
$$\text{thỏa mãn: } p_{zmp, k} \in [zmp_{min}(k), zmp_{max}(k)]$$

Sử dụng thư viện C tối ưu cao **`osqp`** với cơ chế **Warm-Start** để nghiệm số hội tụ chỉ trong $< 1\text{ms}$ trên CPU.

### 3.4. Tầng chấp hành nghịch đảo & Khóa phẳng bàn chân (Kinematics & Leveling)
Từ tọa độ mong muốn của thân và bàn chân trong không gian Descartes:
* **Giải Inverse Kinematics (IK)** cho 6 khớp mỗi chân: `hip_pitch`, `hip_roll`, `hip_yaw`, `knee`, `ankle_pitch`, `ankle_roll`.
* **Khóa phẳng đế bàn chân (Sole Leveling Constraint)**:
  $$q_{ankle\_pitch} = - (q_{hip\_pitch} + q_{knee} + \theta_{pelvis\_pitch})$$
  Đảm bảo đế bàn chân luôn song song với sàn nhà, loại bỏ hiện tượng tiếp đất bằng mũi ngón chân gây lật gót.
* **Chân trụ (Stance Leg)**: Kiểm soát chống lún khớp gối và giữ chiều cao pelvis ($z_{des} = 0.76\text{ m}$).

### 3.5. Giảm chấn tiếp đất & Khóa thế chân rộng (Landing Compliance & Wide Stance)
* Khi chân lăng tiếp đất ($t \ge T_{step}$):
  * **Phanh vận tốc CoM**: Bơm mô-men phanh cổ chân: $\tau_{ankle} \sim K_{brake} \dot{x}_{com}$.
  * **Giảm chấn va chạm (Compliance)**: Khớp gối chân tiếp đất gập nhẹ để triệt tiêu sốc phản lực.
  * **Khóa thế chân rộng (Stable Wide Stance)**: Giữ nguyên vị trí chân bước so le, mở rộng biên đa giác hỗ trợ $zmp_{max}$ trong MPC tương ứng với cự ly bước chân mới, giúp robot dập tắt hoàn toàn năng lượng thừa và đứng vững chãi.

---

## 4. Các file mã nguồn triển khai

```text
controllers/v2_mpc/
├── __init__.py                # Export V2MPCController
├── lipm_model.py              # Mô hình LIPM, ma trận A, B rời rạc & Capture Point (Đã xong)
├── qp_mpc_solver.py           # Bộ giải QP bài toán tối ưu hóa ZMP/CoM (OSQP) (Đã xong)
├── swing_trajectory.py        # Sinh quỹ đạo chân lăng 3D Cycloid minimum-jerk (Đã xong)
├── footstep_planner.py        # FSM quyết định bước chân & tính toán ξ_touchdown (Đã xong)
├── leg_kinematics.py          # Thuật toán Forward & Inverse Kinematics 6-DOF Unitree G1 (Đã xong)
└── mpc_controller.py          # Bộ điều khiển tích hợp hoàn chỉnh (Cần tinh chỉnh pha tiếp đất & an toàn)

benchmark/
└── run_push_benchmark.py      # Mở rộng suite benchmark so sánh 3 bộ: Passive vs V1 PID vs V2 MPC (lực 30N-300N)

scripts/
└── run_interactive.py         # Nâng cấp phím 'M' chuyển đổi trực tiếp controller và demo 220N push step
```

---

## 5. Kế hoạch thực hiện từng bước (Step-by-Step Roadmap)

### Bước 1: Hoàn thiện & Tinh chỉnh `controllers/v2_mpc/mpc_controller.py`
- Kết hợp hoàn hảo giữa In-Place MPC (ZMP QP + Hip Strategy) cho lực $\le 150\text{N}$ và Stepping Recovery (Projected ICP Step + Sole Leveling + Ankle Braking) cho lực $150\text{N} - 300\text{N}+$.
- Kiểm tra tính ổn định trên các mốc lực: $70\text{N}, 120\text{N}, 150\text{N}$ (In-place) và $180\text{N}, 220\text{N}, 260\text{N}, 300\text{N}$ (Stepping Recovery).

### Bước 2: Nâng cấp Benchmark Suite (`benchmark/run_push_benchmark.py`)
- Cấu hình dải lực quét rộng: `[30, 70, 120, 150, 180, 220, 260, 300]` N (thời gian đẩy $0.1\text{s}$).
- Chạy đánh giá đối đầu 3 chiều: **Passive Baseline vs V1 PID vs V2 MPC**.
- Thu thập số liệu và xuất biểu đồ so sánh 4 bảng chuyên nghiệp:
  1. Xung lực tối đa chịu đựng ($J_{max}$ in $\text{N}\cdot\text{s}$ / Force in $\text{N}$).
  2. Góc nghiêng thân tối đa (Max Pelvis Pitch Angle in degrees).
  3. Thời gian dập tắt dao động (Settling Time in seconds).
  4. Quỹ đạo bước chân & Độ dịch chuyển CoM tại các xung lực cực đại.
- Lưu kết quả vào `benchmark/v2_benchmark_results.png`.

### Bước 3: Nâng cấp Interactive 3D Viewer (`scripts/run_interactive.py`)
- Bổ sung phím tắt `M` để chuyển đổi vòng lặp trực tiếp giữa 3 chế độ: `Passive` $\leftrightarrow$ `V1 PID` $\leftrightarrow$ `V2 MPC`.
- Hiển thị HUD thông tin thời gian thực: Chế độ controller hiện tại, Tọa độ Capture Point $\xi_x, \xi_y$, Trạng thái FSM bước chân, Trọng tâm CoM.
- Cập nhật kịch bản Auto-Demo: Trình diễn cú đẩy cực mạnh $220\text{N}$ để người dùng quan sát trực quan robot bước chân đón đầu và phanh đứng vững ở 60 FPS.

### Bước 4: Viết Unit Tests & Tài liệu nghiệm thu (`tests/test_v2_mpc.py`, `walkthrough.md`)
- Xây dựng file test tự động `tests/test_v2_mpc.py` kiểm tra:
  - Thời gian giải QP solver $< 1.5\text{ ms}$.
  - Độ chính xác vị trí chân Inverse Kinematics $< 0.1\text{ mm}$.
  - Chịu lực đẩy $250\text{N}$ qua bước chân phục hồi mà không bị ngã.
- Cập nhật `walkthrough.md` và `README.md` với bảng biểu, số liệu so sánh và biểu đồ kết quả V2.
- Commit toàn bộ thay đổi lên Git.

---

## 6. Tiêu chí nghiệm thu (Verification Plan)

### Automated Tests:
1. `pytest tests/test_v2_mpc.py`: Toàn bộ các bài test QP, IK, và Push Recovery $250\text{N}$ vượt qua $100\%$.
2. `python benchmark/run_push_benchmark.py`: Tạo thành công biểu đồ so sánh 3 bên `benchmark/v2_benchmark_results.png`. V2 MPC phải đạt ngưỡng chịu đựng vượt trội tối thiểu $\ge 260\text{N}$ (so với 150N của V1 và 30N của Passive).

### Manual / Visual Verification:
1. Chạy `python scripts/run_interactive.py` và kiểm tra phím chuyển đổi controller (`M`), các phím đẩy lực (`I, K, J, L`) và Auto-Demo (`P`).
