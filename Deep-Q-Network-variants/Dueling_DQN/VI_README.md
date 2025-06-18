# Dueling Deep Q-Network (Dueling DQN)

## Introduction

Project này là Pytorch minimal implementation [Dueling Deep Q-Network (Dueling_DQN)](https://arxiv.org/pdf/1511.06581). Dueling DQN đề xuất kiến trúc Dueling giúp cải thiện thuật toán Deep Q learning bằng cách tách $Q(s, a)$ (Action-Value Function) thành 2 phần chính $V(s)$ (State-Value function) và $A(s, t)$ (Advantage Function) giúp Model học nhanh, ổn định hơn và tách bạch giá trị của state (theo $V(s)$) và action trên state đó (theo $A(s, t)$) thay gì chỉ dùng $Q(s, a)$ để thể hiện cả 2.

## Algorithm

Cần tìm hiểu `Deep Q-Learning (DQN)` trước. Khi sử dụng DQN, có thể nhận thấy 2 xu hướng:
- Một số state an toàn (nhiều hành động khác nhau đều không tác động nhiều), nên cần quan tâm thêm đến giá trị của state.
- Việc huấn luyện cho từng hành động sẽ làm chậm quá trình huấn luyện, kiến trúc Dueling cho phép tăng tốc huấn luyện.

Cụ thể, tách $Q(s, a)$ thành:
$$
Q(s,a)=V(s)+A(s,a)
$$ 

Dueling DQN chuẩn hóa:
$$
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)
$$

Khi đó khi training cho action $a_1$, các action khác cũng sẽ bị ảnh hưởng bởi lan truyền ngược và Model sẽ được huấn luyện nhanh hơn. Dueling DQN chỉ cần tách model output ra 1 output: n giá trị $Q(s, a_i)$ cho tất cả n action thành 2 outputs, 1 giá trị $V(s)$ và n giá trị $A(s, a_i)$

## Code Structure

Code chạy thử nghiệm Dueling DQN với `**CartPole-v1**` trong [nodebook này](Dueling_DQN.ipynb). Lưu ý: code có thể tồn tại một số bug hoặc không tối ưu!!!

## Trained Model

Bạn có thể load [trained model](trained_model)

## Hyperparameter
Siêu tham số tương tự DQN, bạn cần lưu ý khi tunning vì thuật toán nhạy cảm với siêu tham số:
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

## Result

Dưới đây là kết quả khi train dùng soft update và không dùng.

<p float="left">
  <img src="figure\soft_update.png" alt="soft update" width="500" height="300"/>
  <img src="figure\no_soft_update.png" alt="no soft update" width="500" height="300"/>
</p>

Kết quả model đã đạt tổng phần thưởng tối đa:
- Với cả soft update và không dùng soft update, model nhanh chống đạt tổng phần thưởng tối đa (500) khi test và duy trì mức 500 sau đó.
- Phần thưởng trung bình khi train đều giao động trong khoảng 200-300 và không có dấu hiệu tăng (không quan trọng vì khi train có yếu tố random).
- Environment này rất đơn giản, chỉ để kiểm tra code có hoạt động không nên không thể so sánh 2 cách update hoặc so sánh với DQN, Double DQN.

**Lưu ý**:
- Code có thể tồn tại 1 số bug
- Project có dùng Chatbot để chỉnh lỗi chính tả hoặc format code!

## Reference
- [Dueling DQN paper](https://arxiv.org/pdf/1511.06581)
- [rl-baselines3-zoo hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)
- [DQN paper](https://arxiv.org/pdf/1312.5602)