# Project: Humanoid Balance & Push Recovery

## 1. Idea

Xây dựng một **robot humanoid trong môi trường simulation** có khả năng:

> **Giữ thăng bằng khi đứng và tự phục hồi khi bị tác động bởi lực bên ngoài.**

Ví dụ robot đang đứng thì bị đẩy:

```text
Robot đứng → Bị đẩy → Mất cân bằng → Phản ứng → Phục hồi
```

Project được phát triển trên **cùng một simulation base**, nhưng lần lượt thay đổi phương pháp điều khiển:

```text
V1: PID
      ↓
V2: MPC
      ↓
V3: Reinforcement Learning
      ↓
V4: Hybrid
```

Mục tiêu cuối cùng là **so sánh cách các phương pháp giải quyết cùng một bài toán balance**.

---

## 2. Bài toán chính

Robot humanoid có nhiều joint và trọng tâm cao nên dễ mất thăng bằng.

Khi robot bị đẩy, trạng thái cơ thể thay đổi:

```text
External Force
      ↓
Robot nghiêng / mất cân bằng
      ↓
?
      ↓
Robot phải tạo phản ứng phù hợp
      ↓
Không bị ngã
```

Bài toán đặt ra:

> **Dựa vào trạng thái hiện tại của robot, cần điều khiển các joint như thế nào để robot duy trì hoặc lấy lại thăng bằng?**

---

## 3. Các bài toán con

### State Estimation

> **Robot hiện đang ở trạng thái nào?**

Ví dụ:

* Cơ thể nghiêng bao nhiêu?
* Đang nghiêng theo hướng nào?
* Joint hiện tại ở góc nào?
* Chân nào đang tiếp xúc mặt đất?

---

### Stability Evaluation

> **Robot hiện còn ổn định hay sắp ngã?**

Sử dụng trạng thái cơ thể và trạng thái tiếp xúc mặt đất để đánh giá balance.

---

### Balance Control

> **Robot phải điều chỉnh cơ thể thế nào để không ngã?**

Có thể sử dụng:

* Ankle.
* Knee.
* Hip.
* Torso.
* Nhiều joint cùng lúc.

---

### Push Recovery

> **Khi có ngoại lực làm mất cân bằng, robot phải phản ứng thế nào để phục hồi?**

Tùy mức disturbance:

```text
Push nhỏ
→ điều chỉnh ankle

Push lớn hơn
→ sử dụng hip/body

Push mạnh
→ có thể phải bước chân
```

---

## 4. Các hướng giải quyết

Tất cả cùng giải quyết một bài toán:

> **State → tìm Action → giữ robot cân bằng**

Nhưng theo những cách khác nhau.

### V1 — PID

Robot nhìn vào **sai lệch hiện tại** và liên tục sửa sai.

```text
State → Error → PID → Action
```

### V2 — MPC

Robot sử dụng **model để dự đoán tương lai**, sau đó chọn action tốt.

```text
State → Predict → Optimize → Action
```

### V3 — Reinforcement Learning

Robot **học cách phản ứng thông qua quá trình training**.

```text
State → Policy → Action
```

Policy được học thông qua reward/phạt.

### V4 — Hybrid

Kết hợp Learning và Classical/Optimization Control.

Ví dụ:

```text
State
  ↓
RL Policy
  ↓
Desired Motion
  ↓
MPC / WBC / PID
  ↓
Robot
```

---

## 5. Câu hỏi nghiên cứu của project

Project cuối cùng muốn trả lời:

> **PID, MPC và Reinforcement Learning khác nhau thế nào khi giải quyết cùng bài toán Humanoid Balance & Push Recovery?**

Có thể kiểm chứng bằng các tình huống:

```text
Robot đứng bình thường
Robot bị push nhẹ
Robot bị push vừa
Robot bị push mạnh
```

và quan sát:

> **Robot có ngã không? Phục hồi nhanh thế nào? Phương pháp nào chịu disturbance tốt hơn?**
                  