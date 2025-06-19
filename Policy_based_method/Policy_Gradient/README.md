# policy gradient

## introduction

This project is a pytorch minimal implementation for the `REINFORCE` algorithm, a basic policy gradient algorithm. Policy gradient is an important branch in Reinforcement learning. The goal of the algorithm is to directly train an optimal policy based on the `policy gradient theorem`. REINFORCE is a basic policy gradient algorithm, with 2 variants:
- `REINFORCE` (basic REINFORCE): uses return to calculate loss.
- `REINFORCE with baseline`: uses **advantage(s, a) = return - V(s) = return - baseline** to calculate loss (**V(s)** is called the baseline in the algorithm). baseline helps reduce variance during training.

REINFORCEC is also called monte carlo policy gradient.

## algorithm

### goal of policy gradient

The goal of an RL algorithm is to find an optimal policy $\pi^*$ such that the highest total reward can be achieved from this optimal policy. When parameterizing $\pi$ into a neural network with parameter $\theta$. The goal of RL is:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$
Where:
- $\pi_\theta$ is the policy based on parameter $\theta$, $\pi_\theta(a | s)$ is the probability of choosing action a for state s with policy $\pi_\theta$
- $\tau$ is a trajectory (sequence of state-action)
- $R(\tau)$ is the total reward of trajectory $\tau$

### policy gradient theorem

In the above objective, we already have $\pi_\theta$ (the model being trained), from which an episode (or trajectory) can be generated and the return G of that episode can be calculated. So we need a way to update $\pi_\theta$ based on the trajectory and G.

Based on the policy gradient theorem (it is recommended to read more in-depth materials proving this theory), we need to calculate the gradient $\nabla_\theta J(\theta)$ to update $\theta$:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log (\pi_\theta(a|s)) \cdot Q^{\pi}(s,a) \right]
$$

Once the above gradient is calculated, we can use gradient ascent to update $\theta$.

To implement, we just need to calculate the loss and deep learning libraries will automatically calculate the gradient and update (add negative to turn the problem into gradient descent), when there are N state, action (s, a) pairs (1 episode $\tau$):
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot Q^{\pi}(s,a) )
$$

### REINFORCE

REINFORCE will use return G instead of $Q^{\pi}(s,a)$, after each episode, we will calculate the return for each state in that episode from the terminal state back to the initial state, then there will be N state, action, return to update:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot G(s) )
$$

Because it directly uses return G instead of the expected value (calculated like monte carlo), this algorithm is also called Monte Carlo Policy Gradient.

### REINFORCE with baseline

To reduce variance, we will subtract the baseline from the return. the baseline is the state value function $V(s)$, the loss will become:
$$
loss = - \frac{1}{N} ( \sum_{s, a \in \tau} \log (\pi_\theta(a|s)) \cdot (G(s) - V(s)) )
$$

To calculate V(s), we will create an additional neural network to calculate the value function, update this model using mse between V(s) and label G(s).

## code structure

the code to test REINFORCE with `**CartPole-v1**` is in [this nodebook](REINFORCE\REINFORCE.ipynb), testing REINFORCE with baseline is in [this nodebook](REINFORCE_with_baseline\REINFORCE_with_baseline.ipynb). note: the code may contain some bugs or not be optimal!!!

## trained model

You can load the [REINFORCE trained model](REINFORCE\trained_model) or the [REINFORCE with baseline trained model](REINFORCE_with_baseline\trained_model).

## hyperparameter
- `gamma`: 0.99
- `learning_rate` = 5000 (number of episodes to train)
- `learning_rate` = 2.3e-3
- `test_frequency`: test every `test_frequency` training episodes.
- `num_test_episodes`: test `num_test_episodes` episodes each test.

## result

Each test run may yield different results due to different initial states, so 10 episodes will be tested for objectivity.

### REINFORCE

Below are the REINFORCE results (run 2 times):
- The model can achieve a max of 500 rewards for all 10 episodes during testing.
- However the chart is not stable.

<p float="left">
  <img src="REINFORCE\figure\REINFORCE1.png" width="500" height="300"/>
  <img src="REINFORCE\figure\REINFORCE2.png" width="500" height="300"/>
</p>

### REINFORCE with baseline

Below are the REINFORCE with baseline results (run 2 times), because the model converges very quickly, the first 1000 episodes will be plotted for easier observation:
- The model quickly achieves a max of 500 rewards after about 500 episodes for all 10 episodes during testing and maintains that level throughout the training process.
- Baseline helps the model learn significantly better than basic REINFORCE.

<p float="left">
  <img src="REINFORCE_with_baseline\figure\REINFORCE_with_baseline1.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline\figure\REINFORCE_with_baseline_first1000e1.png" width="500" height="300"/>
</p>

<p float="left">
  <img src="REINFORCE_with_baseline\figure\REINFORCE_with_baseline2.png" width="500" height="300"/>
  <img src="REINFORCE_with_baseline\figure\REINFORCE_with_baseline_first1000e2.png" width="500" height="300"/>
</p>

**note**:
- The code may contain some bugs
- The project used a chatbot to correct spelling errors or format code!