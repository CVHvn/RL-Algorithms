# Policy gradient

## Introduction

This project is a pytorch minimal implementation for the `REINFORCE` algorithm, a basic policy gradient algorithm. Policy gradient is an important branch in reinforcement learning. The goal of the algorithm is to directly train an optimal policy based on the `Policy Gradient Theorem`. `REINFORCE` is a basic policy gradient algorithm, with 2 variants:
- `REINFORCE` (basic REINFORCE): uses return to calculate loss.
- `REINFORCE with baseline`: uses **advantage(s, a) = return - baseline** to calculate loss. Baseline helps reduce variance during training. There are 2 ways to choose a baseline that i found:
    - Baseline = mean return in that episode
    - Baseline = V(s)

`REINFORCE` is also called Monte Carlo Policy Gradient.

## Algorithm

### goal of policy gradient

The goal of the rl algorithm is to find an optimal policy $\pi^*$ such that the highest total reward can be achieved from this optimal policy. When parameterizing $\pi$ into a neural network with parameter $\theta$. The goal of rl is:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$
Where:
- $\pi_\theta$ is the policy based on parameter $\theta$, $\pi_\theta(a | s)$ is the probability of choosing action a for state s with policy $\pi_\theta$
- $\tau$ is a trajectory (sequence of state-action)
- $R(\tau)$ is the total reward of trajectory $\tau$

### Policy Gradient Theorem

In the above objective, we already have $\pi_\theta$ (the model being trained), from which an episode (or trajectory) can be generated and the return G of that episode can be calculated. so we need a way to update $\pi_\theta$ based on the trajectory and G.

Based on the policy gradient theorem (you should consult more in-depth materials for the proof of this theory), we need to calculate the gradient $\nabla_\theta J(\theta)$ to update $\theta$:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log (\pi_\theta(a|s)) \cdot Q^{\pi}(s,a) \right]
$$

Once the above gradient is calculated, we can use gradient ascent to update $\theta$.

To implement, we just need to calculate the loss and deep learning libraries will automatically calculate the gradient and update (add a - sign to turn the problem into gradient descent), when there are n pairs of state, action (s, a) (1 episode $\tau$):
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot Q^{\pi}(s,a) )
$$

### REINFORCE

`REINFORCE` will use return G instead of $Q^{\pi}(s,a)$, after each episode, we will calculate the return for each state in that episode from the terminal state back to the initial state, then there will be n state, action, return to update:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot G(s) )
$$

Because it directly uses return g instead of the expected value (calculated like monte carlo), this algorithm is also called monte carlo policy gradient.

### REINFORCE with baseline

To reduce variance, we will subtract the return by a baseline. the baseline is the mean return in the episode or the state value function $V(s)$, using V(s) the loss becomes:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot (G(s) - V(s)) )
$$

To calculate V(s), we will create an additional neural network to calculate the value function, update this model using mse between V(s) and label G(s).

Using mean return the loss becomes:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot (G(s) - \frac{1}{N} \sum_{i \in 1..n} G_i) )
$$

## Code Structure

Code for testing `REINFORCE` with `**CartPole-v1**` in [this notebook](REINFORCE/REINFORCE.ipynb).
Code for testing `REINFORCE with baseline` as value of state in [this notebook](REINFORCE_with_baseline_value/REINFORCE_with_baseline_value.ipynb).
Code for testing `REINFORCE with baseline` as mean return in episode in [this notebook](REINFORCE_with_baseline_mean_return/REINFORCE_with_baseline_mean_return.ipynb).
Note: the code may contain some bugs or may not be optimal!!!

## Trained Model

You can load [REINFORCE trained model](REINFORCE/trained_model), [REINFORCE with baseline value trained model](REINFORCE_with_baseline_value/trained_model) or [REINFORCE with baseline mean return trained model](REINFORCE_with_baseline_mean_return/trained_model).

## Hyperparameter
- `gamma`: 0.99
- `num_training_episodes` = 5000 (number of episodes to train)
- `learning_rate` = 2.3e-3
- `test_frequency`: test every `test_frequency` training episodes.
- `num_test_episodes`: test `num_test_episodes` episodes each test.

## Result

Each test may yield different results due to different initial states, so 10 episodes will be tested for objectivity.

### REINFORCE

Below is the result of `REINFORCE` with 2 test runs:
- The model can achieve a max of 500 rewards for all 10 episodes when tested.
- However, the chart is unstable.

<p float="left">
  <img src="REINFORCE\figure\REINFORCE1.png" width="500" height="300"/>
  <img src="REINFORCE\figure\REINFORCE2.png" width="500" height="300"/>
</p>

### REINFORCE with baseline

#### Baseline is value of state

Below is the result of `REINFORCE with baseline` with 2 test runs. Since the model converges very quickly, the first 1000 episodes will be plotted for easier observation:
- The model quickly achieves a max of 500 rewards after about 500 episodes for all 10 episodes when tested and maintains that level throughout the training process.
- Baseline helps the model learn significantly better than basic `REINFORCE`.

<p float="left">
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value1.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value_first1000e1.png" width="500" height="300"/>
</p>

<p float="left">
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value2.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_value\figure\REINFORCE_with_baseline_value_first1000e2.png" width="500" height="300"/>
</p>

#### Baseline is mean return

Below is the result of `REINFORCE with baseline` with 2 test runs. since the model converges very quickly, the first 1000 episodes will be plotted for easier observation:
- The model quickly achieves a max of 500 rewards after about 500 episodes for all 10 episodes when tested and maintains that level throughout the training process.
- Baseline helps the model learn significantly better than basic `REINFORCE`.
- This baseline is not as stable as value of state but is easier to code because it doesn't require an additional value network.

<p float="left">
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return1.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return_first1000e1.png" width="500" height="300"/>
</p>

<p float="left">
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return2.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline_mean_return\figure\REINFORCE_with_baseline_mean_return_first1000e2.png" width="500" height="300"/>
</p>

**note**:
- Code may contain some bugs
- This project used a chatbot to correct typos or format code!