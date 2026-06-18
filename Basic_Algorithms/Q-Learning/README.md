# Q-Learning

This folder contains an implementation of the **Q-Learning** algorithm applied to the `Taxi-v4` environment.

---

## Algorithm Overview

**Q-Learning** is an off-policy Temporal Difference (TD) control algorithm used to estimate the value of an optimal policy. Unlike Monte Carlo methods that wait until the end of an episode to update values, Q-Learning updates its action-value function ($Q$-value) at every time step based on the agent's immediate experience.

* **Off-Policy:** (vs. SARSA's On-Policy): The agent learns the optimal $Q$-values independently of the actions taken by the current exploration policy. While SARSA is on-policy and updates its values using the action actually taken next ($A'$), Q-Learning looks ahead at the best possible action ($\max_a Q(S', a)$) in the next state—even if the agent ultimately chooses an exploratory/random step.
* **Temporal Difference Learning:** It updates the $Q$-value using the immediate reward and the maximum estimated value of the next state, removing the need to complete an entire episode before learning.
* **Exploration:** An $\epsilon$-greedy policy is used to balance exploration and exploitation, with $\epsilon$ gradually decaying over time to ensure convergence to the optimal policy.
* **Behavioral Difference**: Because Q-Learning assumes optimal action selection for its updates, it directly learns the optimal policy (path of maximum reward). SARSA, on the other hand, learns a "safer" policy that accounts for the penalties it might incur due to random exploration during training.

---

## Pseudocode

```text
Q-Learning (Off-Policy TD Control) Algorithm:
    Initialize Q(s, a) arbitrarily, and Q(terminal, ·) = 0
    Set α (learning rate), γ (discount factor)
    Set ε ← 1

    Loop for each episode:
        Initialize S
        
        Loop for each step of episode:
            Choose A from S using policy derived from Q (e.g., ε-greedy)
            Take action A, observe R, S'
            
            Q(S, A) ← Q(S, A) + α * [R + γ * max_a Q(S', a) - Q(S, A)]
            
            S ← S'
        Until S is terminal

```

**Q-Value Update Formula:**

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_a Q(S', a) - Q(S, A) \right]$$

**Policy Improvement:**

$$\pi(s) = \begin{cases} \arg\max_a Q(s,a) & \text{with prob } 1 - \varepsilon \\ \text{random} & \text{with prob } \varepsilon \end{cases}$$

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

I use a Jupyter notebook for easier execution. You can use my [notebook](Q-Learning.ipynb) to train or load the [weights](weights/Q-Learning-Taxi-v4-50K-run1-log.pickle) to see the demo.

---

## Experiments & Evaluation

The agent is evaluated using a set of **100 fixed seeds** (in the source configuration) to ensure robust reproducibility and eliminate stochastic bias during testing. Each setup is trained over multiple seeds to capture variability.

### Hyperparameters

You can tune the hyperparameters to see the impact of each setting. For Q-Learning, the learning rate ($\alpha$) and the exploration decay are crucial for smooth convergence:

* `gamma` ($\gamma$) = 0.99
* `alpha` ($\alpha$) = 0.1 (Learning rate)
* `epsilon` ($\epsilon$) = 1.0 (Initial exploration rate)
* `eps_min` = 0.05
* `total_steps` = 50K (Q-Learning converges significantly faster than Monte Carlo, requiring fewer steps).

### Performance Comparison

Below are the evaluation metrics (mean reward, standard deviation, minimum, and maximum rewards across the 100 test seeds) after training for different numbers of steps:

| Steps | Mean Reward | Std | Min Reward | Max Reward |
| --- | --- | --- | --- | --- |
| **1K** | $-93.722$ | $6.597$ | $-107.10$ | $-86.22$ |
| **5K** | $8.620$ | $0$ | $8.62$ | $8.62$ |
| **10K** | $8.620$ | $0$ | $8.62$ | $8.62$ |
| **20K** | $8.620$ | $0$ | $8.62$ | $8.62$ |
| **50K** | $8.620$ | $0$ | $8.62$ | $8.62$ |

**Explanation of metrics:**

* **Mean Reward:** Average return across evaluation runs.
* **Std:** Standard deviation, showing performance stability.
* **Min/Max Reward:** The worst and best-case scenario returns observed during evaluation.

As observed, Q-Learning exhibits extremely fast convergence. By **5K steps**, the agent already achieves the maximum possible mean reward. From that point onwards, the policy has perfectly stabilized at the optimal reward of $8.620$.

### Compare with SARSA

Algorithms convert faster than SARSA, with same config:
* Only need 5K steps instead of 10K to stable yeild maximize reward (8.62). 
* At 1K steps, Q-Learning earn more reward than SARSA (-93.72 > −113).

### Training Chart

Below is the training chart showing the progression of rewards over 1K, 5K, 10K, 20K and 50K training steps:

<div align="center">
<img src="demo\training_chart.png" width="500" alt="training chart" />
</div>

---

## Demo

Here is a demonstration of the trained Q-Learning agent successfully navigating and solving the `Taxi-v4` environment after 50K steps of training:

<div align="center">
<img src="demo\Q-Learning-Taxi-v4-50K-run1.gif" width="400" alt="demo" />
</div>