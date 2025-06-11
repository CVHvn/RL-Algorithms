# Deep Q-Network (DQN)

## Introduction

Project này là Pytorch minimal implementation [Deep Q-Network (DQN)](https://arxiv.org/pdf/1312.5602). `Deep Q-Learning` là 1 thuật toán Reinforcement learning (RL) kết hợp thuật toán `Q-Learning cổ điển` và `Deep Neural Network`. Paper gốc của DeepMind gọi `Deep Q-Learning` là `Deep Q-Network (DQN)` (2 thuật ngữ tương đương nhau). 

## Algorithm

DQN là 1 thuật toán Reinforcement learning (RL) kết hợp thuật toán `Q-Learning cổ điển` và `Deep Neural Network`. Đây là thuật toán off-policy RL, model free RL rất nổi tiếng và là nền tảng của nhiều thuật toán nâng cao sau này.

### Các key ideas:
- Nhờ khả năng của Neural Network, DQN tận dụng NN để dự đoán giá trị Q cho mỗi state thay cho bảng bảng Q-table.
- Cần dùng Target Network (old version của Network đang train) để tính target (Dùng 1 Network để tính target và train là bất khả thi!)
- Dùng Replay Buffer để sample data khi train (tránh overfit!)
- Dùng ε-greedy để sample 1 số action ngẫu nhiên.

### Các bước chính:

- **Khởi tạo:** Model Q (trọng số **θ**), Target Model Q⁻ (trọng số **θ⁻ = θ**), Replay Buffer **D**, done = False

- **Lặp lại mỗi bước huấn luyện:**
    - Chọn action:
        - Với state hiện tại **$s_t$** (reset nếu kết thúc episode, done = True)
        - Chọn hành động **$a_t$** bằng chính sách **ε-greedy** dựa trên Q(sₜ, a; θ):
            - $a_t$ được chọn ngẫu nhiên với xác suất ε
            - ngược lại $a_t = \arg\max_a Q(s_t, a; \theta)$
    - Thực hiện hành động **$a_t$**, nhận phần thưởng **$r_t$**, trạng thái mới **$s_{t+1}$** và done = True hoặc False (True nếu end episode)
    - Lưu bộ dữ liệu **$(s_t, a_t, r_t, s_{t+1})$*$* vào **D**
    - $s_t$ = $s_{t+1}$
    - Sau N step, **Huấn luyện mô hình** 1 hoặc nhiều lần
    - Cập nhật ε (thường là linear decay dựa trên environment step hoặc training step)
    - **Cập nhật mạng mục tiêu:**
        - Gán **θ⁻ ← θ** Sau 1 số step nhất định
        - hoặc **θ⁻ ← θ tau + θ⁻ (1-tau)** mỗi step

- **Huấn luyện mô hình**
   - Lấy minibatch ngẫu nhiên từ **D**
   - Tính giá trị mục tiêu bằng Target Network:
    $
    y_i = 
    \begin{cases}
    r_i + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^-), & \text{if } \texttt{done} = \text{False} \\
    r_i, & \text{otherwise}
    \end{cases}
    $
   - Tính hàm mất mát (loss):
     $L(\theta) = (yᵢ - Q(sᵢ, aᵢ; \theta))²$
   - Cập nhật trọng số mạng Q chính **θ** bằng gradient descent

### Sự khác biệt DQN và Q-Learning cổ điển:
- Hàm Q:
    - Lấy kết quả Q(s, a) từ:
        - DQN dùng Neural Network để dự đoán Q
        - Q-Learning Cổ điển lưu trong bảng
    - Input, Output:
        - Để tiện cho việc triển khai, Neural Network sẽ nhận vào state $s_t$ và trả và N giá trị $Q_a$ cho mỗi action a. 
        - Bảng Q sẽ nhận vào input là cả state $s_t$ và action $a_t$ và trả về 1 giá trị $Q_{s, a}$
    - Ưu điểm DQN:
        - DQN dùng được cho không gian trạng thái (state-space) rất lớn hoặc vô hạn, bảng Q không thể lưu hoặc lưu cũng 0 tính được cho từng $Q_{s, a}$.
        - DQN có thể predict tốt cho các state, action ít hoặc không được train nhờ sự tổng quát của NN. Bảng Q không thể vì chỉ lưu $Q_{s, a}$ cho các state được train.
- Cách cập nhật:
    - Target Network:
        - Vì dùng NN hiện tại để vừa tính target vừa train là bất khả thi nên cần dùng target network để tính target.
        - Q-learning cổ điển chỉ dùng 1 bảng Q.
    - Replay Buffer:
        - Vì NN dễ overfit nếu chỉ train dựa theo episode hiện tại, cần dùng Replay Buffer để sample dữ liệu khi training.
        - Q-learning không cần (vẫn có thể dùng Replay Buffer nếu bạn thích!)

| Thuộc tính             | Q-Learning Cổ điển                         | Deep Q-Learning (DQN)                           |
|------------------------|--------------------------------------------|-------------------------------------------------|
| **Hàm Q**              | Bảng tra Q(s, a)                           | Mạng nơ-ron xấp xỉ Q(s, a; θ)                   |
| **Không gian trạng thái** | Nhỏ, rời rạc                            | Lớn hoặc liên tục (ảnh, cảm biến,...)          |
| **Khả năng tổng quát** | Gần như không có                           | Cao nhờ học đặc trưng qua DNN                   |
| **Replay Buffer**      | Không sử dụng                              | Có – giúp ổn định huấn luyện                    |
| **Target Network**     | Không có                                   | Có – tránh cập nhật không ổn định               |
| **Cần Deep Learning?** | Không                                      | Có                                              |

## Code Structure

Code chạy thử nghiệm MCTS với `**CartPole-v1**` trong [nodebook này](DQN.ipynb). Lưu ý: code có thể tồn tại một số bug hoặc không tối ưu!!!

## Trained Model

Bạn có thể load [trained model](trained_model)

## Hyperparameter
Vì DQN nhạy cảm với siêu tham số, tôi tham khảo hyperparameter tại [rl-baselines3-zoo hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml) để đảm bảo code hoạt động!. Họ không để tham số mặc định trong file và số step hoặc số epoch train không để rõ ràng nên tôi gán là 500000:
- `gamma`: 0.99
- `batch_size`: = 64
- `buffer_size` = 100000
- `total_steps` = 500000 (số environment step khi training)
- `start_training_step` = 1000 (train từ environment step nào)
- `learning_rate` = 2.3e-3
- `train_frequency`= 256 (train model sau mỗi `train_frequency` environment step)
- `epochs` = 128 (mỗi lần train model sẽ train 128 epoch)
- `update_frequency` = 10 (cập nhật target model sau `update_frequency` environment step) hoặc `tau` = 0.005 (soft update mỗi environment step)
- `epsilon` (ε):
    - `init`: 1
    - `end`: 0.04
    - Linear decay với `exploration_fraction` = 0.16 (16% `total_steps` đầu)

**Lưu ý**: DQN cũng như nhiều thuật toán Deep RL khác đều rất nhạy cảm với siêu tham số, bạn có thể thử tunning, nếu đặt siêu tham số không phù hợp thì model gần như chỉ random hoặc chỉ đạt được ít phần thưởng khi test! Cần cẩn thận khi tuning `total_steps`, `buffer_size`, `gamma`, `learning_rate`, `train_frequency`, `epochs`, `ε`!

## Result

Dưới đây là kết quả khi train dùng soft update và không dùng.

<p float="left">
  <img src="figure\soft update.png" alt="soft update" width="500" height="300"/>
  <img src="figure\no soft update.png" alt="no soft update" width="500" height="300"/>
</p>

Kết quả model đã đạt tổng phần thưởng tối đa:
- Với cả soft update và không dùng soft update, model nhanh chống đạt tổng phần thưởng tối đa (500) khi test và duy trì mức 500 sau đó.
- Phần thưởng trung bình khi train đều giao động trong khoảng 200-300 và không có dấu hiệu tăng (không quan trọng vì khi train có yếu tố random).
- Environment này rất đơn giản, chỉ để kiểm tra code có hoạt động không nên không thể so sánh 2 cách.

**Lưu ý**:
- Code có thể tồn tại 1 số bug
- Project có dùng Chatbot để chỉnh lỗi chính tả hoặc format code!

## Reference
- [DQN paper](https://arxiv.org/pdf/1312.5602)
- [rl-baselines3-zoo hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)