# Locomotion

## 1. Locomotion Overview

### 1.1. Locomotion là gì?

**Locomotion** là bài toán giúp robot tạo ra chuyển động của cơ thể để
di chuyển trong môi trường.

Ví dụ: - Humanoid sử dụng chân để đứng, đi, chạy và giữ thăng bằng. -
Quadruped phối hợp bốn chân để di chuyển. - Mobile robot sử dụng bánh xe
để di chuyển.


**Humanoid** là robot có cơ thể và cách vận động mô phỏng con người.

### 1.2. Các thuật ngữ nền tảng

  
  ----------------------------------- -----------------------------------
  **Joint**                           Khớp chuyển động của robot, ví dụ
                                      khớp gối

  **Link**                            Phần cứng nối giữa các khớp, ví dụ
                                      cẳng chân

  **DOF**                             Số hướng/chuyển động độc lập mà
                                      robot có thể thực hiện

  **Actuator**                        Bộ phận tạo chuyển động, thường là
                                      motor

  **Gait**                            Kiểu hoặc chu kỳ bước đi của robot

  **Balance**                         Khả năng giữ thăng bằng, không bị
                                      ngã

  **Contact**                         Điểm cơ thể robot đang tiếp xúc với
                                      môi trường, ví dụ bàn chân với mặt
                                      đất

  **Torque**                          Lực xoắn mà motor tạo ra tại khớp

  **Controller**                      Điều khiển robot để chuyển động
                                      thực tế giống chuyển động mong muốn

  **Feedback**                        Thông tin robot đo lại để biết
                                      chuyển động thực tế có đúng với
                                      mong muốn hay không

### 1.3. Các bài toán chính của Locomotion

  -----------------------------------------------------------------------
  Bài toán                            Mục tiêu
  ----------------------------------- -----------------------------------
  **Balance / Stability**             Giữ robot thăng bằng và không bị
                                      ngã

  **Gait Generation**                 Tạo kiểu và chu kỳ bước đi

  **Footstep Planning**               Quyết định robot nên đặt chân ở đâu

  **Trajectory Generation**           Tạo chuyển động của chân/joint theo
                                      thời gian

  **Whole-Body Motion**               Phối hợp chuyển động của chân, tay
                                      và thân

  **Locomotion Control**              Điều khiển robot thực hiện chuyển
                                      động mong muốn
  -----------------------------------------------------------------------

### 1.4. Các hướng tiếp cận Locomotion

  ------------------------------------------------------------------------
  Hướng                    Ý tưởng đơn giản        Thuật ngữ thường gặp
  ------------------------ ----------------------- -----------------------
  **Classical /            Dùng mô hình toán và    Kinematics, Dynamics,
  Model-based**            vật lý của robot để     PID, ZMP
                           tính chuyển động        

  **Optimization-based**   Tìm chuyển động tối ưu  MPC, Trajectory
                           nhưng vẫn thỏa các ràng Optimization,
                           buộc                    Whole-Body Control

  **Learning-based**       Cho robot học cách di   Reinforcement Learning,
                           chuyển từ quá trình     Imitation Learning,
                           training hoặc dữ liệu   Policy

  **Hybrid**               Kết hợp                 RL + MPC/WBC, Learned
                           model/optimization với  Policy + Controller
                           learning                


## 2. Balance / Stability

### 2.1. Balance / Stability là gì?

**Balance / Stability** là bài toán giúp robot giữ thăng bằng và không
bị ngã khi đứng, khi di chuyển hoặc khi chịu tác động từ bên ngoài.

Ví dụ: một humanoid đang đứng và bị đẩy.

``` text
Robot bị đẩy
    ↓
Cơ thể bị nghiêng
    ↓
Phát hiện mất cân bằng
    ↓
Tính cách lấy lại cân bằng
    ↓
Điều chỉnh joint / motor
    ↓
Robot lấy lại thăng bằng
```

> Balance = Làm sao để robot không bị ngã?

### 2.2. Quy trình của bài toán Balance

Có thể chia bài toán Balance thành bốn phần chính:

  ----------------------------------- -----------------------------------
  **State Estimation**                Robot hiện đang nghiêng bao nhiêu,
                                      chuyển động thế nào và chân nào
                                      đang chạm đất?

  **Stability Evaluation**            Trạng thái hiện tại có ổn định hay
                                      robot đang có nguy cơ ngã?

  **Balance Planning**                Robot nên nghiêng người, thay đổi
                                      lực hay bước chân như thế nào để
                                      lấy lại cân bằng?

  **Balance Control**                 Điều khiển joint/motor như thế nào
                                      để thực hiện chuyển động đã tính?

Pipeline tổng quát:

``` text
Sensors
   ↓
State Estimation
   ↓
Stability Evaluation
   ↓
Balance Planning
   ↓
Balance Control
   ↓
Joint / Motor
   ↓
Robot
   ↓
Feedback
   └────────────→ quay lại State Estimation
```

### 2.3. State Estimation

Trước khi giữ thăng bằng, robot phải biết trạng thái hiện tại của chính
mình.

Robot có thể cần biết: 
- Cơ thể đang nghiêng bao nhiêu. 
- Các joint đang
ở vị trí nào. 
- Robot đang chuyển động với tốc độ nào. 
- Chân nào đang
tiếp xúc với mặt đất. 
- Lực tác dụng lên từng chân.

Một số sensor thường được sử dụng: 
- **IMU:** đo gia tốc và tốc độ
quay. 
- **Joint Encoder:** đo vị trí/góc của joint. - 
**Force/Torque Sensor:** đo lực và torque. 
- **Contact Sensor:** xác định chân có đang
tiếp xúc với mặt đất hay không.

``` text
Sensors
   ↓
State Estimation
   ↓
Current Robot State
```

### 2.4. Stability Evaluation

Sau khi biết trạng thái hiện tại, robot cần xác định mình có đang cân
bằng hay không.

Một số khái niệm thường gặp:

**Center of Mass (CoM):** điểm đại diện cho vị trí tập trung khối lượng
của toàn bộ robot.

**Support Polygon:** vùng được tạo bởi các điểm/bề mặt mà robot đang
tiếp xúc với mặt đất, ví dụ vùng hỗ trợ tạo bởi hai bàn chân.

**ZMP (Zero Moment Point):** một khái niệm thường được sử dụng trong
điều khiển robot chân để đánh giá và thiết kế trạng thái cân bằng động.

Ở mức tổng quan, có thể hiểu rằng hệ thống sử dụng trạng thái robot và
các đại lượng liên quan để đánh giá robot còn ổn định hay đang có nguy
cơ mất cân bằng.

### 2.5. Balance Planning

Nếu phát hiện robot mất cân bằng, hệ thống phải quyết định cách lấy lại
cân bằng.

Ba chiến lược cơ bản:

**Ankle Strategy:** điều chỉnh cổ chân khi mức mất cân bằng còn nhỏ.

**Hip Strategy:** sử dụng hông và thân trên khi cần phản ứng mạnh hơn.

**Stepping Strategy:** bước một chân sang vị trí mới để tạo vùng hỗ trợ
mới khi robot không thể lấy lại cân bằng chỉ bằng ankle hoặc hip.

``` text
Mất cân bằng nhỏ
       ↓
Ankle Strategy

Mất cân bằng lớn hơn
       ↓
Hip Strategy

Có nguy cơ ngã
       ↓
Stepping Strategy
```

### 2.6. Balance Control

Sau khi xác định chuyển động mong muốn, controller phải biến nó thành
lệnh cho các joint và actuator.

``` text
Desired Motion
      ↓
Controller
      ↓
Joint / Motor Commands
      ↓
Physical Robot
      ↓
Sensor Feedback
```

Controller liên tục sử dụng feedback để điều chỉnh sai lệch giữa chuyển
động mong muốn và chuyển động thực tế.

### 2.7. Classical / Model-based Approach

Hướng Classical / Model-based sử dụng mô hình toán học và vật lý của
robot để xác định trạng thái cân bằng và tính cách điều khiển robot.

Pipeline đơn giản:

``` text
Sensors
   ↓
Robot State
   ↓
Physical / Mathematical Model
   ↓
Balance Calculation
   ↓
Controller
   ↓
Joint / Motor
```

Các thuật ngữ thường gặp: 
- **Kinematics** 
- **Dynamics** 
- **ZMP** 
- **Inverted Pendulum Model** 
- **PID** 
- **Inverse Dynamics**

**Inverted Pendulum Model** có thể hiểu đơn giản là mô hình hóa robot
giống một con lắc ngược cần được điều khiển liên tục để không bị đổ.

### 2.8. Optimization-based Approach

Hướng Optimization-based đặt bài toán giữ thăng bằng dưới dạng một bài
toán tối ưu.

Thay vì thiết kế từng luật phản ứng cụ thể, hệ thống tìm chuyển động tốt
nhất trong khi phải thỏa nhiều điều kiện, ví dụ: 
- Robot không bị ngã. 
- Đi đúng hướng. 
- Không vượt giới hạn joint. 
- Không sử dụng lực/torque vượt giới hạn.

Một kỹ thuật quan trọng là **MPC (Model Predictive Control)**.

MPC sử dụng mô hình để dự đoán trạng thái trong tương lai, lựa chọn điều
khiển phù hợp, thực hiện một phần rồi tiếp tục tính lại.

``` text
Current State
      ↓
Predict Future
      ↓
Optimization
      ↓
Best Control
      ↓
Robot
      ↓
Measure New State
      ↓
Repeat
```

> **MPC = nhìn trước → chọn điều khiển tốt → thực hiện → đo lại → tính
> lại.**

Một thuật ngữ khác thường gặp là **Whole-Body Control (WBC)**, dùng để
phối hợp nhiều bộ phận của robot như chân, hông, thân và tay nhằm thực
hiện đồng thời nhiều mục tiêu.

### 2.9. Learning-based Approach

Hướng Learning-based cho robot học cách phản ứng từ dữ liệu hoặc quá
trình training.

Một hướng phổ biến là **Reinforcement Learning (RL)**.

``` text
Robot State / Observation
          ↓
       RL Policy
          ↓
        Action
          ↓
    Joint / Motor
          ↓
        Robot
```

Trong simulation, robot có thể trải qua nhiều tình huống như bị nghiêng,
bị đẩy hoặc di chuyển trên các bề mặt khác nhau.

Thông qua reward, policy dần học được hành động giúp robot duy trì hoặc
lấy lại thăng bằng.

Ví dụ:

``` text
Giữ được thăng bằng → Reward +
Đi đúng mục tiêu     → Reward +
Bị ngã               → Reward -
```

> **Learning-based Balance = robot học cách phản ứng để duy trì cân
> bằng.**

### 2.10. Hybrid Approach

Hybrid kết hợp Learning-based với Model-based hoặc Optimization-based.

Ví dụ:

``` text
RL Policy
   ↓
Desired Motion
   ↓
WBC / Controller
   ↓
Joint / Motor
   ↓
Robot
```

Trong kiến trúc này, learned policy có thể quyết định chuyển động mong
muốn, còn controller phía dưới đảm nhiệm việc thực hiện chuyển động trên
robot.

Một cách tổng quát:

``` text
Balance
│
├── Classical / Model-based
│   └── ZMP, Inverted Pendulum, PID, Dynamics
│
├── Optimization-based
│   └── MPC, WBC, Trajectory Optimization
│
├── Learning-based
│   └── RL, Policy
│
└── Hybrid
    └── Learning + Model/Optimization-based Control
```
