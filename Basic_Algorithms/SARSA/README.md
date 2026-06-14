# SARSA

This folder contains an implementation of the **SARSA** algorithm applied to the `Taxi-v4` environment.

---

## Algorithm Overview

**SARSA (State-Action-Reward-State-Action)** is an on-policy Temporal Difference (TD) control algorithm used to estimate the value of a policy. Unlike Monte Carlo methods that wait until the end of an episode to update values, SARSA updates its action-value function ($Q$-value) at every time step based on the agent's immediate experience.

* **On-Policy:** The agent learns the $Q$-values based on the actual actions taken by the current policy $\pi$, including the exploration steps.
* **Temporal Difference Learning:** It updates the $Q$-value using the immediate reward and the estimated value of the next state-action pair, removing the need to complete an entire episode before learning.
* **Exploration:** An $\epsilon$-greedy policy is used to balance exploration and exploitation, with $\epsilon$ gradually decaying over time to ensure convergence to the optimal policy.

---

## Pseudocode

```text
SARSA (On-Policy TD Control) Algorithm:
    Initialize Q(s, a) arbitrarily, and Q(terminal, ·) = 0
    Set α (learning rate), γ (discount factor)
    Set ε ← 1

    Loop for each episode:
        Initialize S
        Choose A from S using policy derived from Q (e.g., ε-greedy)
        
        Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g., ε-greedy)
            
            Q(S, A) ← Q(S, A) + α * [R + γ * Q(S', A') - Q(S, A)]
            
            S ← S'
            A ← A'
        Until S is terminal

```

**Q-Value Update Formula:**

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

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

I use a Jupyter notebook for easier execution. You can use my [notebook](SARSA.ipynb) to train or load the [weights](weights/SARSA-Taxi-v4-50K-run1-log.pickle) to see the demo.

---

## Experiments & Evaluation

The agent is evaluated using a set of **100 fixed seeds** (in the source configuration) to ensure robust reproducibility and eliminate stochastic bias during testing. Each setup is trained over multiple seeds to capture variability.

### Hyperparameters

You can tune the hyperparameters to see the impact of each setting. For SARSA, the learning rate ($\alpha$) and the exploration decay are crucial for smooth convergence:

* `gamma` ($\gamma$) = 0.99
* `alpha` ($\alpha$) = 0.1 (Learning rate)
* `epsilon` ($\epsilon$) = 1.0 (Initial exploration rate)
* `eps_min` = 0.05
* `total_steps` = 50K (SARSA converges significantly faster than Monte Carlo, requiring fewer steps).

### Performance Comparison

Below are the evaluation metrics (mean reward, standard deviation, minimum, and maximum rewards across the 100 test seeds) after training for different numbers of steps:

| Steps | Mean Reward | Std | Min Reward | Max Reward |
| --- | --- | --- | --- | --- |
| **1K** | $-113.070$ | $8.609$ | $-123.95$ | $-96.76$ |
| **5K** | $6.908$ | $1.999$ | $2.42$ | $8.60$ |
| **10K** | $8.578$ | $0.033$ | $8.52$ | $8.62$ |
| **20K** | $8.620$ | $0.000$ | $8.62$ | $8.62$ |
| **50K** | $8.620$ | $0.000$ | $8.62$ | $8.62$ |

**Explanation of metrics:**

* **Mean Reward:** Average return across evaluation runs.
* **Std:** Standard deviation, showing performance stability.
* **Min/Max Reward:** The worst and best-case scenario returns observed during evaluation.

As observed, SARSA exhibits extremely fast convergence. By **5K steps**, the agent already achieves a positive mean reward. From **20K steps** onwards, the standard deviation drops to virtually zero ($1.77 \times 10^{-15}$), meaning the policy has perfectly stabilized at the optimal reward of $8.620$.

### Training Chart

Below is the training chart showing the progression of rewards over 1K, 5K, 10K, 20K and 50K training steps:

<div align="center">
<img src="demo\training_chart.png" width="500" alt="training chart" />
</div>

---

## Demo

Here is a demonstration of the trained SARSA agent successfully navigating and solving the `Taxi-v4` environment after 50K steps of training:

<div align="center">
<img src="demo\SARSA-Taxi-v4-50K-run1.gif" width="400" alt="demo" />
</div>