# Dyna-Q

This folder contains an implementation of the **Dyna-Q** algorithm applied to the `Taxi-v4` environment.

---

## Algorithm Overview

**Dyna-Q** is a model-based reinforcement learning algorithm that integrates online learning, model learning, and planning. It combines the direct model-free approach of Q-Learning with a learned model of the environment to accelerate convergence and improve sample efficiency.

* **Integration of Learning and Planning:** The agent uses its real experiences to update its action-value function ($Q$-value) via standard Q-Learning (direct RL) and simultaneously trains an environment model.
* **Environment Model:** Use another table to store the environment model. Save $Model_{s,a}=(s', r)$ every real environment step and use this table to create planning data by simulating the next state and reward from $Model_{s,a}$ instead of obtaining them directly from the real environment.
* **Planning Steps:** At each time step, after updating from real data, the agent performs a specified number of planning steps ($n$). It randomly samples previously visited states and actions from its internal model, simulates the next states and rewards, and updates the $Q$-values using these simulated experiences.
* **Temporal Difference Learning:** Both direct learning and planning updates leverage the standard TD control mechanism, removing the need to complete an entire episode before updating values.
* **Exploration:** An $\epsilon$-greedy policy is used to balance exploration and exploitation, with $\epsilon$ gradually decaying over time to ensure convergence to the optimal policy.

---

## Pseudocode

```text
Dyna-Q Algorithm (Integrating Learning, Modeling, and Planning):
    Initialize Q(s, a) and Model(s, a) for all s ∈ S, a ∈ A
    Set α (learning rate), γ (discount factor)
    Set ε ← 1, n (number of planning steps)

    Loop for each episode:
        Initialize S
        
        Loop for each step of episode:
            Choose A from S using policy derived from Q (e.g., ε-greedy)
            Take action A, observe R, S'
            
            Direct RL Update:
            Q(S, A) ← Q(S, A) + α * [R + γ * max_a Q(S', a) - Q(S, A)]
            
            Model Learning:
            Model(S, A) ← R, S' (assuming a deterministic environment)
            
            Planning Loop (repeat n times):
                S_sim ← random previously visited state
                A_sim ← random action previously taken in S_sim
                R_sim, S'_sim ← Model(S_sim, A_sim)
                Q(S_sim, A_sim) ← Q(S_sim, A_sim) + α * [R_sim + γ * max_a Q(S'_sim, a) - Q(S_sim, A_sim)]
            
            S ← S'
        Until S is terminal

```

**Q-Value Update Formula (Used in both Direct RL and Planning):**

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

I use a Jupyter notebook for easier execution. You can use my [notebook](Dyna-Q.ipynb) to train or load the [weights](weights/Dyna-Q-Taxi-v4-run1-log.pickle) to see the demo.

---

## Experiments & Evaluation

The agent is evaluated using a set of **100 fixed seeds** (in the source configuration) to ensure robust reproducibility and eliminate stochastic bias during testing. Each setup is trained over multiple seeds to capture variability.

### Hyperparameters

You can tune the hyperparameters to see the impact of each setting. For Dyna-Q, the number of planning steps ($n$) interacts dynamically with the total training episodes:

* `gamma` ($\gamma$) = 0.99
* `alpha` ($\alpha$) = 0.1 (Learning rate)
* `epsilon` ($\epsilon$) = 1.0 (Initial exploration rate)
* `eps_min` = 0.05
* `repeat_times` = 10 (Num planning step update per env step)

### Performance Comparison

Below are the evaluation metrics (mean reward, standard deviation, minimum, and maximum rewards across the 100 test seeds) after training with different episode configurations and planning steps:

| Episodes | Mean Reward | Std | Min Reward | Max Reward |
| --- | --- | --- | --- | --- |
| **100-10** | $-137.070$ | $21.795$ | $-191.18$ | $-111.47$ |
| **100-100** | $-25.100$ | $31.771$ | $-109.65$ | $0.36$ |
| **100-500** | $2.618$ | $8.448$ | $-14.87$ | $8.60$ |
| **500-10** | $5.348$ | $2.909$ | $-1.58$ | $8.62$ |
| **500-100** | $8.618$ | $0.006$ | $8.60$ | $8.62$ |
| **500-500** | $8.602$ | $0.035$ | $8.50$ | $8.62$ |
| **1K-10** | $8.620$ | $0$ | $8.62$ | $8.62$ |

**Explanation of metrics:**

* **Episodes Suffix (`-number`):** = repeat_times (*500-100* mean train 500 episode with repeat_times = 100).
* **Mean Reward:** Average return across evaluation runs.
* **Std:** Standard deviation, showing performance stability.
* **Min/Max Reward:** The worst and best-case scenario returns observed during evaluation.

As observed, Dyna-Q exhibits extremely fast convergence when utilizing planning steps. By introducing planning, the agent achieves near-optimal performance much earlier in terms of environmental interaction episodes compared to base learning.

Dyna-Q can yeild higher rewards by increase repeat_times from 10 to 500. Even with 100 episodes, We can get positive rewards with Dyna-Q. Dyna-Q just need 500 episodes to get near optimal solution and just 1K episodes to get maximum possible mean reward (8.62).

### Effect of Planning Steps

Planning steps significantly accelerate the learning process compared to pure model-free learning:

* At **100 episodes**, increasing the planning steps from 0 to 500 brings the mean reward from $-137.070$ up to $2.618$.
* At **500 episodes**, adding just 100 planning steps allows the agent to reach a near-optimal mean reward of $8.618$, effectively stabilizing the policy with fewer total environmental episodes.
* At **1K episodes**, we can get maximum possible mean reward (8.62) while normal Q-Learning just get $-93.722$ rewards.

### Training Chart

Below is the training chart showing the progression of rewards over different training episodes and planning configurations:

<div align="center">
<img src="demo\training_chart.png" width="500" alt="training chart" />
</div>

---

## Demo

Here is a demonstration of the trained Dyna-Q agent successfully navigating and solving the `Taxi-v4` environment:

<div align="center">
<img src="demo\Dyna-Q-Taxi-v4-1K-run1.gif" width="400" alt="demo" />
</div>