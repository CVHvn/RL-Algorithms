# Policy gradient

## Introduction

Project này là Pytorch minimal implementation cho thuật toán `REINFORCE`, 1 thuật toán policy gradient cơ bản. Policy gradient là 1 nhánh quan trọng trong Reinforcement learning. Mục tiêu của thuật toán là huấn luyện trực tiếp 1 chính sách (policy) tối ưu dựa theo `Policy Gradient Theorem`. REINFORCE là 1 thuật toán policy gradient cơ bản, có 2 biến thể:
- `REINFORCE` (REINFORCE cơ bản): dùng return để tính loss.
- `REINFORCE with baseline`: dùng **advantage(s, a) = return - V(s) = return - baseline** để tính loss (**V(s)** được gọi là baseline trong thuật toán). Baseline giúp giảm phương sai khi training. 

REINFORCEC cũng được gọi là Monte Carlo Policy Gradient.

## Algorithm

### Mục tiêu của policy gradient

Mục tiêu của thuật toán RL là tìm được một policy $\pi^*$ tối ưu sao cho có thể đạt được tổng phần thưởng cao nhất từ policy tối ưu này. Khi tham số hóa $\pi$ thành Neural Network có tham số $\theta$. Mục tiêu của RL là:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$
Với:
- $\pi_\theta$ là policy dựa theo tham số $\theta$, $\pi_\theta(a | s)$ là xác suất chọn hành động a cho state s với policy $\pi_\theta$
- $\tau$ là một trajectory (chuỗi trạng thái-hành động)
- $R(\tau)$ là tổng phần thưởng của trajectory $\tau$

### Policy Gradient Theorem

Trong mục tiêu trên, ta đã có $\pi_\theta$ (model đang train), từ đó sẽ tạo ra được episode (hay trajectory) và có thể tính toán được return G của episode đó. Vậy chúng ta cần 1 cách để cập nhật $\pi_\theta$ dựa theo trajectory và G.

Dựa vào Policy Gradient Theorem (nên tìm hiểu các tài liệu chuyên sâu hơn chứng minh lý thuyết này), ta cần tính gradient $\nabla_\theta J(\theta)$ để cập nhật $\theta$:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log (\pi_\theta(a|s)) \cdot Q^{\pi}(s,a) \right]
$$

Khi tính được gradient trên, ta có thể dùng gradient ascent để cập nhật $\theta$. 

Để implement, ta chỉ cần tính loss và các thư viện học sâu sẽ tự tính gradient và cập nhật (thêm dấu - để biến bài toán thành gradient descent), khi có N cặp state, action (s, a) (1 episode $\tau$):
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot Q^{\pi}(s,a) )
$$

### REINFORCE

REINFORCE sẽ dùng return G thay cho $Q^{\pi}(s,a)$, sau mỗi episode, ta sẽ tính return cho mỗi state trong episode đó từ terminal state về state ban đầu, khi đó sẽ có N state, action, return để cập nhật:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot G(s) )
$$

Vì sử dụng trực tiếp return G thay cho giá trị kì vọng (được tính như Monte Carlo), thuật toán này còn được gọi là Monte Carlo Policy Gradient.

### REINFORCE with baseline

Để giảm phương sai, ta sẽ trừ return cho baseline. Baseline là state value function $V(s)$, loss sẽ trở thành:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot (G(s) - V(s)) )
$$

Để tính V(s), ta sẽ tạo thêm 1 Neural Network để tính value function, cập nhật Model này bằng mse giữa V(s) và label G(s).

## Code Structure

Code chạy thử nghiệm REINFORCE với `**CartPole-v1**` trong [nodebook này](REINFORCE\REINFORCE.ipynb), thử nghiệm REINFORCE with baseline trong [nodebook này](REINFORCE_with_baseline\REINFORCE_with_baseline.ipynb). Lưu ý: code có thể tồn tại một số bug hoặc không tối ưu!!!

## Trained Model

Bạn có thể load [REINFORCE trained model](REINFORCE\trained_model) hoặc [REINFORCE with baseline trained model](REINFORCE_with_baseline\trained_model).

## Hyperparameter
- `gamma`: 0.99
- `learning_rate` = 5000 (số episode để train)
- `learning_rate` = 2.3e-3
- `test_frequency`: test mỗi `test_frequency` training episodes.
- `num_test_episodes`: test `num_test_episodes` episodes mỗi lần test.

## Result

Mỗi lần test có thể cho kết quả khác nhau vì init state khác nhau, nên sẽ test 10 episodes để khách quan. Mình chỉ chạy code 1 lần cho 1 environment đơn giản, có thể khi chạy nhiều hơn với seed khác nhau sẽ cho kết quả khác!

### REINFORCE

Dưới đây là kết quả REINFORCE:
- Model có thể đạt max 500 rewards cho cả 10 episodes khi test.
- Tuy nhiên chart không được ổn định lắm.

<p float="left">
  <img src="REINFORCE\figure\REINFORCE.png" width="500" height="300"/>
</p>

### REINFORCE with baseline

Dưới đây là kết quả REINFORCE with baseline, Vì model hội tựu rất nhanh nên sẽ plot 1000 episode đầu cho dễ quan sát:
- Model nhanh chống đạt max 500 rewards sau khoảng 500 episodes cho cả 10 episodes khi test và duy trì mức đó đến hết quá trình training.
- baseline giúp model học tốt hơn hẳn REINFORCE cơ bản.

<p float="left">
  <img src="REINFORCE_with_baseline\figure\REINFORCE_with_baseline.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline\figure\REINFORCE_with_baseline_first1000e.png" width="500" height="300"/>
</p>

**Lưu ý**:
- Code có thể tồn tại 1 số bug
- Project có dùng Chatbot để chỉnh lỗi chính tả hoặc format code!