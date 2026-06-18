# Double Q-Learning

This folder contains an implementation of the **Double Q-Learning** algorithm applied to the `Taxi-v4` environment.

---

## Algorithm Overview

**Double Q-Learning** is an off-policy Temporal Difference (TD) control algorithm designed to mitigate the **maximization bias** inherent in standard Q-Learning. Standard Q-Learning uses the same Q-table both to select the best action and to estimate its value, which often leads to systematically overestimating action values, especially in noisy environments.

* **Two Estimators ($Q_A$ and $Q_B$):** Double Q-Learning maintains two independent action-value functions, $Q_A$ and $Q_B$. Each step, one Q-table is randomly chosen to be updated. It uses its own weights to determine the maximizing action but uses the other Q-table's weights to estimate the value of that action.
* **Off-Policy:** Like standard Q-Learning, the agent learns the optimal policy independently of the exploratory actions taken during training.
* **Temporal Difference Learning:** It updates the value functions using immediate rewards and bootstrapping from subsequent states without waiting for the episode to end.
* **Exploration:** An $\epsilon$-greedy policy is used based on the average or sum of both Q-tables ($Q_A + Q_B$) to balance exploration and exploitation.

---

## Pseudocode

```text
Double Q-Learning (Off-Policy TD Control) Algorithm:
    Initialize QA(s, a) and QB(s, a) arbitrarily, and QA(terminal, ·) = QB(terminal, ·) = 0
    Set α (learning rate), γ (discount factor)
    Set ε ← 1

    Loop for each episode:
        Initialize S
        
        Loop for each step of episode:
            Choose A from S using policy derived from QA and QB (e.g., ε-greedy via QA + QB)
            Take action A, observe R, S'
            
            With probability 0.5:
                A* = argmax_a QA(S', a)
                QA(S, A) ← QA(S, A) + α * [R + γ * QB(S', A*) - QA(S, A)]
            Else:
                A* = argmax_a QB(S', a)
                QB(S, A) ← QB(S, A) + α * [R + γ * QA(S', A*) - QB(S, A)]
            
            S ← S'
        Until S is terminal

```

**Q-Value Update Formulas:**

With 0.5 probability (updating $Q_A$ using $Q_B$ for evaluation):


$$Q_A(S, A) \leftarrow Q_A(S, A) + \alpha \left[ R + \gamma Q_B\left(S', \arg\max_a Q_A(S', a)\right) - Q_A(S, A) \right]$$

Otherwise (updating $Q_B$ using $Q_A$ for evaluation):


$$Q_B(S, A) \leftarrow Q_B(S, A) + \alpha \left[ R + \gamma Q_A\left(S', \arg\max_a Q_B(S', a)\right) - Q_B(S, A) \right]$$

**Policy Improvement:**

$$\pi(s) = \begin{cases} \arg\max_a \left( Q_A(s,a) + Q_B(s,a) \right) & \text{with prob } 1 - \varepsilon \\ \text{random} & \text{with prob } \varepsilon \end{cases}$$

---

## Environment: Taxi-v4

The **Taxi** environment involves a $5 \times 5$ grid world. There are 4 designated locations marked as R, G, Y, and B. A passenger starts at one of these locations and wishes to be dropped off at another. The taxi starts at a random square.

* **State Space:** 500 discrete states ($25$ taxi positions $\times$ $5$ passenger locations $\times$ $4$ destination locations).
* **Action Space:** 6 discrete actions:
* `0`: Move South (South)
* `1`: Move North (North)
* `2`: Move East (East)
* `3`: Move West (West)
* `4`: Pickup passenger (Pickup)
* `5`: Drop off passenger (Drop off)


* **Rewards:**
* $-1$ per step.
* $+20$ for a successful drop-off.
* $-10$ for illegal pickup/drop-off actions.



---

## Code

I use a Jupyter notebook for easier execution. You can use my [notebook](Double-Q-Learning.ipynb) to train or load the [weights](weights/Double-Q-Learning-Taxi-v4-50K-run1-log.pickle) to see the demo.

---

## Experiments & Evaluation

The agent is evaluated using a set of **100 fixed seeds** (in the source configuration) to ensure robust reproducibility and eliminate stochastic bias during testing. Each setup is trained over multiple seeds to capture variability.

### Hyperparameters

The hyperparameters configured for Double Q-Learning are as follows:

* `gamma` ($\gamma$) = 0.99
* `alpha` ($\alpha$) = 0.1 (Learning rate)
* `epsilon` ($\epsilon$) = 1.0 (Initial exploration rate)
* `eps_min` = 0.05
* `total_steps` = 50K

### Performance Comparison

Below are the evaluation metrics (mean reward, standard deviation, minimum, and maximum rewards across the 100 test seeds) after training for different numbers of steps:

| Steps | Mean Reward | Std | Min Reward | Max Reward |
| --- | --- | --- | --- | --- |
| **1K** | $-175.94$ | $5.29$ | $-187.2$ | $-170.18$ |
| **5K** | $6.18$ | $2.38$ | $0.47$ | $8.62$ |
| **10K** | $8.62$ | $0$ | $8.62$ | $8.62$ |
| **20K** | $8.62$ | $0$ | $8.62$ | $8.62$ |
| **50K** | $8.62$ | $0$ | $8.62$ | $8.62$ |

**Explanation of metrics:**

* **Mean Reward:** Average return across evaluation runs.
* **Std:** Standard deviation, showing performance stability.
* **Min/Max Reward:** The worst and best-case scenario returns observed during evaluation.

As observed from the table, Double Q-Learning demonstrates strong convergence characteristics. Because it decoupling action selection from value evaluation using two separate Q-tables, it requires a few more exploratory steps initially compared to standard Q-Learning. However, by **10K steps**, the agent completely stabilizes, yielding the maximum optimal mean reward of $8.62$.

### Compare with Standard Q-Learning & SARSA

* **Convergence Speed:** Double Q-Learning takes slightly more steps to fully stabilize (10K steps) than standard Q-Learning (5K steps) because it splits updates across two independent estimators. However, it still converges faster or on par with traditional SARSA architectures.
* **Stability:** Once converged at 10K steps, Double Q-Learning perfectly reaches the absolute optimal reward of 8.62 without variance, eliminating any potential overestimation bias.

### Training Chart

Below is the training chart showing the progression of rewards over 1K, 5K, 10K, 20K and 50K training steps:

<div align="center">
<img src="demo\training_chart.png" width="500" alt="training chart" />
</div>

---

## Demo

Here is a demonstration of the trained Double Q-Learning agent successfully navigating and solving the `Taxi-v4` environment after 50K steps of training:

<div align="center">
<img src="demo\Double-Q-Learning-Taxi-v4-50K-run1.gif" width="400" alt="demo" />
</div>