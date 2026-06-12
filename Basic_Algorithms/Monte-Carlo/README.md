# Monte Carlo

This folder contains an implementation of the **Every Visit Monte Carlo** algorithm applied to the `Taxi-v4` environment.

---

## Algorithm Overview

**Monte Carlo (MC)** is a model-free reinforcement learning algorithm that learns optimal policies directly from episodes of experience. MC have two version: **First-Visit** and **Every-Visit**. I use **Every-Visit** in this test.
* **On-Policy:** The agent evaluates and improves the same policy that it uses to make decisions.
* **First-Visit:** The return ($G_t$) is accumulated and averaged only for the first time a state-action pair $(s, a)$ is visited within an episode.
* **Every-Visit:** The return ($G_t$) is accumulated and averaged for the any time a state-action pair $(s, a)$ is visited within an episode.
* **Q-Value** pair $(s, a)$ is calculate by mean of return ($G_{s a}$)
* **Exploration:** An $\epsilon$-greedy policy is used to balance exploration and exploitation, with $\epsilon$ gradually decaying over time to ensure convergence to the optimal policy.

---

## Pseudocode

```pseudocode
Monte Carlo Every Visit Algorithm:
    Initialize Q(s, a) = 0, Returns(s, a) = {} for all s ∈ S, a ∈ A
    Set ε ← 1, k ← 1

    Loop forever:
        Generate episode (s₁,a₁,r₁, ..., s_T,a_T,r_T) under π
        For t = 1, ..., T:
            G_t ← cumulative return from step t  [see formula below]
            Append G_t to Returns(s_t, a_t)
            Q(s_t, a_t) ← mean(Returns(s_t, a_t))
        k ← k + 1
        ε ← max(1/k, ε_min)
        π ← ε-greedy(Q)

    Return Q, π
```

**Return formula:**

$G_t = \sum_{j=t}^{T} r_{kj}$

**Policy improvement:**

$\pi(s) = \begin{cases} \arg\max_a Q(s,a) & \text{with prob } 1 - \varepsilon \\ \text{random} & \text{with prob } \varepsilon \end{cases}$

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

I use jupyter notebook for easier run. You can use my [notebook](Monte_Carlo.ipynb) to train or load [weight](weights/MC-Taxi-v4-1M-run1.pickle) to see demo.

---

## Experiments & Evaluation

The agent is evaluated using a set of **100 fixed seeds** (in the source configuration) to ensure robust reproducibility and eliminate stochastic bias during testing. Each setup is trained 10 seeds to capture variability.

### Hyperparameter

You can tune hyperparameter to see the impact of each hyperparameter. In my opinion, gamma isn't important, the number of episodes, epsilon and decay_step are the more important hyperparameters:
  * `gamma` = 0.99.
  * `epsilon` = 1
  * `eps_min` = 0.05
  * `total_episode` = 1e6, MC need long training to converging.
  * `decay_step` = 90% `total_episode`, MC need long decay step to explore envirment.

### Performance Comparison

Below are the evaluation metrics (mean reward, standard deviation, minimum, and maximum rewards across the 100 test seeds) after training for different numbers of episodes:

| Episodes | Mean Reward | Std | Min Reward | Max Reward |
| :--- | :---: | :---: | :---: | :---: |
| **20K** | $-89.83$ | $17.05$ | $-113.99$ | $-65.95$ |
| **100K** | $-44.09$ | $21.19$ | $-88.62$ | $-22.37$ |
| **1M** | $2.43$ | $8.61$ | $-14.10$ | $8.62$ |

Explain:
  * Mean Reward: Average of 10 runs (each run is the average of 100 episode seeds)
  * Std: std of 10 runs (each run is the average of 100 episode seeds)
  * Min Reward: min of 10 runs (each run is the average of 100 episode seeds)
  * Max Reward: min of 10 runs (each run is the average of 100 episode seeds)

As observed, performance improves dramatically as the number of episodes increases. By **1M episodes**, the agent successfully achieves a positive mean reward, indicating that it has converged toward an efficient, successful policy.

### Training Chart

Below is a chart showing the average last 100 trains and test episode rewards when running 20K, 100K, and 1M steps.

<div align="center">
<img src="demo\training_chart.png" width="1000" alt="training chart" />
</div>

---

## Demo

Here is a demonstration of the trained Monte Carlo agent interacting with the `Taxi-v4` environment after 1 million episodes of training:

<div align="center">
<img src="demo\MC-Taxi-v4-1M-run1.gif" width="400" alt="demo" />
</div>