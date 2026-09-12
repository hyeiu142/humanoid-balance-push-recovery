# Implementation Plan: Humanoid Balance & Push Recovery (V1 - Baseline PID)

Tài liệu thiết kế chi tiết triển khai bài toán **Humanoid Balance & Push Recovery** trong môi trường mô phỏng **MuJoCo** với robot **Unitree G1**, bắt đầu từ phiên bản **V1: Virtual Model Control (VMC) / Task-Space PID** làm nền tảng so sánh cho các phiên bản tiếp theo (V2 MPC, V3 RL, V4 Hybrid).

---

## 1. Kiến trúc hệ thống & Thiết kế tổng thể

Hệ thống được thiết kế theo dạng module hóa với giao diện chuẩn để đảm bảo **cùng một simulation base** có thể cắm rút linh hoạt các controller khác nhau (V1 PID, V2 MPC, V3 RL, V4 Hybrid):

```text
               ┌────────────────────────────────────────────────────────┐
               │                     MuJoCo Physics                     │
               │             (Unitree G1 MJCF Menagerie)                │
               └──────────────┬──────────────────────────▲──────────────┘
                              │ Sensors / State          │ Actuator Torques
                              ▼                          │
               ┌─────────────────────────────┐           │
               │       State Estimator       │           │
               │ - CoM pos/vel               │           │
               │ - Torso Roll/Pitch/Yaw & ω  │           │
               │ - Joint pos/vel             │           │
               │ - Foot Contacts & CoP / ZMP │           │
               └──────────────┬──────────────┘           │
                              │ RobotState               │
                              ▼                          │
               ┌─────────────────────────────┐           │
               │      Controller Module      │           │
               │   (V1: VMC Task-Space PID)  ├───────────┘
               │  - Ankle Strategy (Small)   │
               │  - Hip Strategy (Medium)    │
               │  - Stance Posture PD        │
               └─────────────────────────────┘
```

---

## 2. Các thành phần kỹ thuật chi tiết

### 2.1. Quản lý môi trường & Mô hình Robot
- **Môi trường Python**: Sử dụng `uv venv` và cài đặt `mujoco`, `numpy`, `scipy`, `matplotlib`.
- **Mô hình Robot**: Tải trực tiếp bộ model chính thức của **Unitree G1** từ [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1):
  - Model XML (`g1.xml`), meshes (`assets/`), scene setup (`scene.xml`).
  - 12 khớp chân hoạt động (6 DOF mỗi chân: `hip_pitch`, `hip_roll`, `hip_yaw`, `knee`, `ankle_pitch`, `ankle_roll`).
  - Cố định (lock/stiff PD) các khớp thân trên (arms, waist) ở tư thế đứng chuẩn để tập trung phân tích phản ứng của chân.

### 2.2. State Estimator & Stability Evaluator (`sim/sensors.py`)
Trích xuất và tính toán các đại lượng vật lý cốt lõi:
1. **Trọng tâm (Center of Mass - CoM)**: Vị trí $p_{com} = [x, y, z]^T$ và vận tốc $v_{com}$.
2. **Góc nghiêng thân (Torso Attitude)**: Roll $\phi$, Pitch $\theta$, Yaw $\psi$ và vận tốc góc $\omega$ từ cảm biến IMU gắn tại pelvis/torso.
3. **Áp lực tiếp xúc & Điểm đặt lực (Center of Pressure - CoP / ZMP)**: Tính từ lực pháp tuyến tại 4 điểm tiếp xúc (geoms) dưới mỗi bàn chân.
4. **Support Polygon (Đa giác hỗ trợ)**: Xác định biên giới hạn bàn chân để đánh giá xem robot còn trong vùng ổn định hay đang lật (tipping limit).

### 2.3. Bộ điều khiển V1: Virtual Model Control PID (`controllers/v1_pid/`)
Sử dụng phương pháp **Virtual Model Control (Pratt et al.)**:
1. **Virtual Components**:
   - **Thẳng đứng (Torso Pitch/Roll PID)**: Đặt lò xo & giảm chấn ảo tại thân để sinh ra mô-men ảo $\tau_{pitch}^{des}, \tau_{roll}^{des}$ kéo thân robot về phương thẳng đứng khi bị nghiêng.
   - **Độ cao (CoM Height PID)**: Đặt lò xo ảo theo trục Z để giữ thân ở độ cao danh định $z_{des} \approx 0.75\text{m}$.
2. **Chiến lược Ankle & Hip (Ankle & Hip Strategies)**:
   - **Ankle Strategy (Đẩy nhẹ)**: Khi độ lệch nhỏ ($|\theta| < \theta_{thresh}$), mô-men ảo được truyền trực tiếp xuống `ankle_pitch` và `ankle_roll` để dịch chuyển CoP nhằm đẩy CoM về vị trí cân bằng mà không gập hông.
   - **Hip Strategy (Đẩy vừa)**: Khi độ lệch lớn hơn hoặc CoP tiến sát mép bàn chân, mô-men ảo kích hoạt gập khớp `hip_pitch` và `hip_roll` để tạo mô-men quán tính ngược chiều, giúp giữ bàn chân phẳng với mặt đất.
3. **Stance Posture PD**: Duy trì vị trí danh định cho các khớp đầu gối và hông yaw để chống sụp gối.
4. **Torque Saturation**: Giới hạn lực mô-men đầu ra theo đúng thông số phần cứng thực tế của Unitree G1.

### 2.4. Công cụ kiểm thử & Đánh giá (Dual Mode)
1. **Chế độ tương tác 3D (`scripts/run_interactive.py`)**:
   - Mở cửa sổ 3D của MuJoCo Viewer.
   - Cho phép người dùng tác động ngoại lực (push disturbance) tức thời qua bàn phím:
     - Phím mũi tên (Lên/Xuống: Đẩy Tới/Lui; Trái/Phải: Đẩy Ngang).
     - Phím số 1, 2, 3: Chọn cường độ lực (Nhẹ 30N, Vừa 80N, Mạnh 150N).
   - Hiển thị trực tiếp vector lực và trạng thái CoM/CoP trên cửa sổ 3D.
2. **Chế độ Benchmark tự động (`benchmark/run_push_benchmark.py`)**:
   - Chạy mô phỏng không cần mở GUI (headless) để quét tự động dải xung lực:
     - Lực: $F \in [10\text{N}, 20\text{N}, ..., 200\text{N}]$, thời gian tác động $\Delta t = 0.1\text{s}$.
     - Hướng tác động: Forward ($+x$), Backward ($-x$), Lateral ($+y$).
   - Ghi nhận và xuất biểu đồ/bảng số liệu:
     - **Ngưỡng xung lực tối đa ($J_{max}$)** trước khi robot ngã.
     - **Thời gian ổn định trở lại ($t_{settle}$)** về trạng thái đứng thẳng ($|\text{tilt}| < 1^\circ$).
     - **Độ lệch cực đại của CoM và CoP**.

---

## 3. Cấu trúc thư mục dự kiến

```text
/home/yennguyen/vr/
├── .venv/                         # Python virtual environment (uv)
├── docs/
│   ├── baitoan.md
│   └── locomotion_balance.md
├── models/
│   └── unitree_g1/                # MJCF, meshes, textures từ MuJoCo Menagerie
│       ├── g1.xml
│       ├── scene.xml
│       └── assets/
├── sim/
│   ├── __init__.py
│   ├── simulation_base.py         # Lớp quản lý vòng lặp MuJoCo, reset, step
│   ├── sensors.py                 # State estimation (CoM, CoP, IMU, Contacts)
│   └── disturbance.py             # Cơ chế tiêm ngoại lực đẩy (impulse/step push)
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py        # Interface chung cho tất cả các controller
│   └── v1_pid/
│       ├── __init__.py
│       ├── vmc_pid_controller.py  # Virtual Model Control (Torso/CoM to Leg torques)
│       └── balance_strategies.py  # Logic phối hợp Ankle vs Hip strategy
├── benchmark/
│   ├── __init__.py
│   ├── metrics.py                 # Tính settling time, max impulse, fall detection
│   └── run_push_benchmark.py      # Script chạy quét lực tự động và vẽ biểu đồ
├── scripts/
│   ├── setup_env.sh               # Script tạo venv và tải model Unitree G1
│   └── run_interactive.py         # Demo tương tác trực tiếp với MuJoCo Viewer
└── README.md                      # Hướng dẫn chạy và kết quả benchmark
```

---

## 4. Kế hoạch thực hiện (Step-by-step Execution)

### Bước 1: Khởi tạo môi trường & Tải mô hình
- Khởi tạo virtual environment bằng `uv`.
- Cài đặt `mujoco`, `numpy`, `scipy`, `matplotlib`.
- Tải mô hình chuẩn của `unitree_g1` từ `mujoco_menagerie` vào `models/unitree_g1/`.

### Bước 2: Xây dựng Simulation Base & State Estimator
- Viết `sim/simulation_base.py` để load model, khởi tạo trạng thái đứng ổn định ban đầu (nominal standing posture).
- Viết `sim/sensors.py` tính CoM, CoP, IMU orientation/angular velocity, và phát hiện tiếp xúc bàn chân.
- Viết `sim/disturbance.py` tạo lực đẩy tại thân (pelvis/torso) với biên độ và thời gian chỉ định.

### Bước 3: Phát triển Controller V1 (VMC Task-Space PID)
- Xây dựng `controllers/base_controller.py`.
- Xây dựng `controllers/v1_pid/vmc_pid_controller.py`:
  - Vòng lặp Torso Roll/Pitch PID.
  - Phân bổ mô-men cho Ankle pitch/roll và Hip pitch/roll.
  - Giữ độ cao và dáng đứng bằng posture PD.
- Tinh chỉnh thông số $K_p, K_d$ để robot đứng vững tĩnh 10s không bị trôi hoặc rung lắc.

### Bước 4: Tích hợp Đẩy & Phản xạ Ankle/Hip
- Tinh chỉnh Ankle Strategy khi bị đẩy nhẹ (20N-40N, 0.1s): Bàn chân đứng yên, cổ chân bù lực.
- Tinh chỉnh Hip Strategy khi bị đẩy vừa (50N-90N, 0.1s): Thân gập để triệt tiêu mô-men quán tính.
- Xác định điểm giới hạn: Khi đẩy > 100N, robot ngã do vượt quá Support Polygon (chứng minh giới hạn của giải thuật tại chỗ).

### Bước 5: Hoàn thiện Interactive Viewer & Benchmark Suite
- Viết `scripts/run_interactive.py` với phím bấm để người dùng tương tác trực quan.
- Viết `benchmark/run_push_benchmark.py` xuất bảng chỉ số và đồ thị phản ứng (settling time, CoM trajectory).

---

## 5. Kế hoạch kiểm chứng (Verification Plan)

### Kiểm chứng tự động (Automated Tests):
1. **Kiểm tra môi trường**: Chạy script python import `mujoco` và load thành công `models/unitree_g1/scene.xml`.
2. **Kiểm tra trạng thái đứng tĩnh (Standing Test)**: Chạy 5000 bước mô phỏng (5 giây) không có ngoại lực: CoM dao động $< 1\text{cm}$, góc nghiêng thân $< 0.5^\circ$.
3. **Kiểm tra phản ứng đẩy nhẹ (Ankle Recovery)**: Tác dụng lực $30\text{N}$ trong $0.1\text{s}$ theo chiều tới ($+x$): robot phục hồi về vị trí đứng thẳng trong vòng $< 1.5\text{s}$, hai bàn chân không bị nhấc khỏi mặt đất.
4. **Kiểm tra phản ứng đẩy vừa (Hip Recovery)**: Tác dụng lực $70\text{N}$ trong $0.1\text{s}$: robot gập thân, không bị lật ngã, phục hồi trong $< 2.5\text{s}$.
5. **Chạy Benchmark tự động**: Chạy `python benchmark/run_push_benchmark.py` sinh ra kết quả bảng thống kê và file biểu đồ `.png`.

### Kiểm chứng trực quan (Manual / Visual Verification):
- Chạy `python scripts/run_interactive.py`, bấm các phím mũi tên để quan sát phản ứng của Unitree G1 trên cửa sổ MuJoCo 3D Viewer.
