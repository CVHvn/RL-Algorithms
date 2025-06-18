# Dueling Deep Q-Network (Dueling DQN)

## Introduction

This project is a pytorch minimal implementation of [Dueling Deep Q-Network (Dueling_DQN)](https://arxiv.org/pdf/1511.06581). Dueling DQN proposes the Dueling architecture which improves the Deep Q-learning algorithm by splitting $Q(s, a)$ (Action-Value Function) into 2 main parts: $V(s)$ (State-Value function) and $A(s, a)$ (Advantage Function), helping the Model learn faster, be more stable, and separate the value of the state (via $V(s)$) from the action on that state (via $A(s, a)$) instead of just using $Q(s, a)$ to represent both.

## Algorithm

You should understand `Deep Q-Learning (DQN)` first. When using DQN, two trends can be observed:
- Some states are safe (many different actions do not have a significant impact), so it's necessary to pay more attention to the value of the state.
- Training for each action will slow down the training process; the Dueling architecture allows for faster training.

Specifically, splitting $Q(s, a)$ into:
$$
Q(s,a)=V(s)+A(s,a)
$$ 

Dueling DQN standardizes this as:
$$
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)
$$

Then, when training for action $a_1$, other actions will also be affected by backpropagation, and the Model will be trained faster. Dueling DQN only needs to change the model output from a single output (n values $Q(s, a_i)$ for all n actions) to 2 outputs: one value $V(s)$ and n values $A(s, a_i)$.

## Code Structure

The code for testing Dueling DQN with `**CartPole-v1**` in [this notebook](Dueling_DQN.ipynb). Note: the code may contain some bugs or may not be optimal!!!

## Trained Model

You can load the [trained model](trained_model).

## Hyperparameter
Hyperparameters are similar to DQN; you need to be careful when tunning as the algorithm is sensitive to hyperparameters:
- `gamma`: 0.99
- `batch_size`: = 64
- `buffer_size` = 100000
- `total_steps` = 500000 (number of environment steps during training)
- `start_training_step` = 1000 (which environment step to start training from)
- `learning_rate` = 2.3e-3
- `train_frequency`= 256 (train the model every `train_frequency` environment steps)
- `epochs` = 128 (each time the model is trained, it will train for 128 epochs)
- `update_frequency` = 10 (update the target model after `update_frequency` environment steps) or `tau` = 0.005 (soft update every environment step)
- `epsilon` (ε):
    - `init`: 1
    - `end`: 0.04
    - Linear decay with `exploration_fraction` = 0.16 (the first 16% of `total_steps`)

## Result

Below are the results when training with and without soft update.

<p float="left">
  <img src="figure\soft_update.png" alt="soft update" width="500" height="300"/>
  <img src="figure\no_soft_update.png" alt="no soft update" width="500" height="300"/>
</p>

The resulting model has reached the maximum total reward:
- With both soft update and no soft update, the model quickly reaches the maximum total reward (500) during testing and maintains the 500 level afterwards.
- The average reward during training fluctuates between 200-300 and shows no sign of increasing (this is not important as there is a random factor during training).
- This environment is very simple, just to check if the code works, so it's not possible to compare the two update methods or compare with DQN, Double DQN.

**Note**:
- The code may contain some bugs
- This project used a Chatbot to fix typos or format the code!

## Reference
- [Dueling DQN paper](https://arxiv.org/pdf/1511.06581)
- [rl-baselines3-zoo hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)
- [DQN paper](https://arxiv.org/pdf/1312.5602)