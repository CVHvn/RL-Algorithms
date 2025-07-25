# Policy gradient

## Introduction

Project này là Pytorch minimal implementation cho thuật toán `REINFORCE`, 1 thuật toán policy gradient cơ bản. Policy gradient là 1 nhánh quan trọng trong Reinforcement learning. Mục tiêu của thuật toán là huấn luyện trực tiếp 1 chính sách (policy) tối ưu dựa theo `Policy Gradient Theorem`. REINFORCE là 1 thuật toán policy gradient cơ bản, có 2 biến thể:
- `REINFORCE` (REINFORCE cơ bản): dùng return để tính loss.
- `REINFORCE with baseline`: dùng **advantage(s, a) = return - baseline** để tính loss. Baseline giúp giảm phương sai khi training. Có 2 cách chọn baseline mà mình tìm thấy:
    - baseline = mean return trong episode đó
    - baseline = V(s)

REINFORCE cũng được gọi là Monte Carlo Policy Gradient.

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

Để giảm phương sai, ta sẽ trừ return cho baseline. Baseline là mean return trong episode hoặc state value function $V(s)$, dùng V(s) thì loss sẽ trở thành:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot (G(s) - V(s)) )
$$

Để tính V(s), ta sẽ tạo thêm 1 Neural Network để tính value function, cập nhật Model này bằng mse giữa V(s) và label G(s).

Dùng mean return thì loss trở thành:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot (G(s) - \frac{1}{N} \sum_{i \in 1..n} G_i) )
$$

## Code Structure

Code chạy thử nghiệm REINFORCE với `**CartPole-v1**` trong [nodebook này](REINFORCE/REINFORCE.ipynb).
Code thử nghiệm REINFORCE with baseline là value of state trong [nodebook này](REINFORCE_with_baseline_value/REINFORCE_with_baseline_value.ipynb).
Code thử nghiệm REINFORCE with baseline là mean return trong episode trong [nodebook này](REINFORCE_with_baseline_mean_return/REINFORCE_with_baseline_mean_return.ipynb).
Lưu ý: code có thể tồn tại một số bug hoặc không tối ưu!!!

## Trained Model

Bạn có thể load [REINFORCE trained model](REINFORCE/trained_model), [REINFORCE with baseline value trained model](REINFORCE_with_baseline_value/trained_model) hoặc [REINFORCE with baseline mean return trained model](REINFORCE_with_baseline_mean_return/trained_model).

## Hyperparameter
- `gamma`: 0.99
- `learning_rate` = 5000 (số episode để train)
- `learning_rate` = 2.3e-3
- `test_frequency`: test mỗi `test_frequency` training episodes.
- `num_test_episodes`: test `num_test_episodes` episodes mỗi lần test.

## Result

Mỗi lần test có thể cho kết quả khác nhau vì init state khác nhau, nên sẽ test 10 episodes để khách quan. 

### REINFORCE

Dưới đây là kết quả REINFORCE với 2 lần chạy thử:
- Model có thể đạt max 500 rewards cho cả 10 episodes khi test.
- Tuy nhiên chart không được ổn định lắm.

<p float="left">
  <img src="REINFORCE\figure\REINFORCE1.png" width="500" height="300"/>
  <img src="REINFORCE\figure\REINFORCE2.png" width="500" height="300"/>
</p>

### REINFORCE with baseline

#### Baseline is value of state

Dưới đây là kết quả REINFORCE with baseline với 2 lần chạy thử, Vì model hội tựu rất nhanh nên sẽ plot 1000 episode đầu cho dễ quan sát:
- Model nhanh chống đạt max 500 rewards sau khoảng 500 episodes cho cả 10 episodes khi test và duy trì mức đó đến hết quá trình training.
- baseline giúp model học tốt hơn hẳn REINFORCE cơ bản.

<p float="left">
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value1.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value_first1000e1.png" width="500" height="300"/>
</p>

<p float="left">
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value2.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value_first1000e2.png" width="500" height="300"/>
</p>

#### Baseline is mean return

Dưới đây là kết quả REINFORCE with baseline với 2 lần chạy thử, Vì model hội tựu rất nhanh nên sẽ plot 1000 episode đầu cho dễ quan sát:
- Model nhanh chống đạt max 500 rewards sau khoảng 500 episodes cho cả 10 episodes khi test và duy trì mức đó đến hết quá trình training.
- Baseline giúp model học tốt hơn hẳn REINFORCE cơ bản.
- Baseline này không ổn định bằng value of state nhưng dễ code hơn vì không cần thêm Value Network

<p float="left">
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return1.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return_first1000e1.png" width="500" height="300"/>
</p>

<p float="left">
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return2.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return_first1000e2.png" width="500" height="300"/>
</p>

**Lưu ý**:
- Code có thể tồn tại 1 số bug
- Project có dùng Chatbot để chỉnh lỗi chính tả hoặc format code!