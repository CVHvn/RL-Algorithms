# Deep Q-Network (DQN)

## Introduction

This project is a minimal PyTorch implementation of the [Deep Q-Network (DQN)](https://arxiv.org/pdf/1312.5602). `Deep Q-Learning` is a Reinforcement Learning (RL) algorithm that combines the classic `Q-Learning` algorithm with `Deep Neural Networks`. The original paper by DeepMind refers to `Deep Q-Learning` as `Deep Q-Network (DQN)` (the two terms are equivalent).

## Algorithm

DQN is a well-known off-policy, model-free Reinforcement Learning (RL) algorithm that combines the classic `Q-Learning` algorithm with `Deep Neural Networks`. It serves as a foundation for many advanced algorithms that followed.

### Key Ideas:
- Leveraging the power of Neural Networks, DQN uses an NN to predict Q-values for each state instead of a Q-table.
- A **Target Network** (an older version of the network being trained) is necessary to calculate the target values. (Using a single network to both calculate the target and be trained is impossible!)
- A **Replay Buffer** is used to sample data during training (to prevent overfitting!).
- An **ε-greedy** policy is used to sample some random actions for exploration.

### Main Steps:

- **Initialization:** Initialize Q-network Q (with weights **θ**), Target Network Q⁻ (with weights **θ⁻ = θ**), Replay Buffer **D**, and set done = False.

- **Loop for each step:**
    - **Choose an action:**
        - Given the current state **$s_t$** (reset if an episode ends, done = True).
        - Select an action **$a_t$** using an **ε-greedy** policy based on Q(sₜ, a; θ):
            - With probability ε, **$a_t$** is chosen randomly.
            - Otherwise, $a_t = \arg\max_a Q(s_t, a; θ)$.
    - Execute action **$a_t$**, observe the reward **$r_t$**, the new state **$s_{t+1}$**, and whether the episode has ended (`done`).
    - Store the transition **$(s_t, a_t, r_t, s_{t+1}, \text{done})$** in the replay buffer **D**.
    - Set $s_t \leftarrow s_{t+1}$.
    - After N steps, **train the model** one or more times.
    - Update ε (usually with a linear decay based on environment steps or training steps).
    - **Update the target network:**
        - Hard update: Set **θ⁻ ← θ** after a fixed number of steps.
        - or Soft update: **θ⁻ ← τθ + (1-τ)θ⁻** at each step.

- **Model Training:**
   - Sample a random minibatch of transitions from **D**.
   - Calculate the target value using the Target Network:
    $
    y_i = 
    \begin{cases}
    r_i + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^-), & \text{if } \texttt{done} = \text{False} \\
    r_i, & \text{otherwise}
    \end{cases}
    $
   - Calculate the loss:
     $L(\theta) = (y_i - Q(s_i, a_i; \theta))^2$
   - Update the weights **θ** of the main Q-network using gradient descent.

### Differences Between DQN and Classic Q-Learning:
- **Q-Function:**
    - How Q(s, a) is obtained:
        - DQN uses a Neural Network to predict Q-values.
        - Classic Q-Learning stores them in a table.
    - Input/Output:
        - For implementation convenience, the Neural Network takes a state $s_t$ as input and outputs N Q-values, one for each action a.
        - The Q-table takes both the state $s_t$ and action $a_t$ as input and returns a single value $Q_{s, a}$.
    - Advantages of DQN:
        - DQN can handle very large or continuous state spaces, where a Q-table would be too large to store or compute.
        - DQN can generalize to predict reasonable Q-values for states or actions that were rarely or never seen during training. A Q-table cannot do this.
- **Update Mechanism:**
    - **Target Network:**
        - Using the current network to both calculate the target and be updated is unstable. DQN uses a separate target network to stabilize training.
        - Classic Q-learning uses only one Q-table.
    - **Replay Buffer:**
        - Neural Networks can easily overfit if trained on sequential data from the current episode. A Replay Buffer is used to sample diverse data for training.
        - Q-learning doesn't strictly require a Replay Buffer (but you could use this if you wanted!).

| Property               | Classic Q-Learning                       | Deep Q-Learning (DQN)                       |
|------------------------|------------------------------------------|---------------------------------------------|
| **Q-Function**         | Lookup table Q(s, a)                     | Neural network approximates Q(s, a; θ)      |
| **State Space**        | Small, discrete                          | Large or continuous (images, sensors, ...)  |
| **Generalization**     | Almost none                              | High, by learning features via a DNN        |
| **Replay Buffer**      | Not used                                 | Yes – stabilizes training                   |
| **Target Network**     | None                                     | Yes – prevents unstable updates             |
| **Requires Deep Learning?** | No                                  | Yes                                         |

## Code Structure

The code for testing DQN with `**CartPole-v1**` is available in [this notebook](DQN.ipynb). Note: the code may contain some bugs or may not be fully optimized!

## Trained Model

You can load the [trained model](trained_model).

## Hyperparameters
DQN is sensitive to hyperparameters. For this project, I referred to the hyperparameters from the [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml) to ensure the code works. They don't specify default values in the file, and the number of training steps/epochs is not clearly defined, so I set `total_steps` to 500,000:
- `gamma`: 0.99
- `batch_size`: 64
- `buffer_size`: 100,000
- `total_steps`: 500,000 (number of environment steps during training)
- `start_training_step`: 1,000 (environment step to start training from)
- `learning_rate`: 2.3e-3
- `train_frequency`: 256 (train the model every `train_frequency` environment steps)
- `epochs`: 128 (train for 128 epochs each training time)
- `update_frequency`: 10 (update target model every `update_frequency` environment steps) or `tau` = 0.005 (for soft updates every environment step)
- `epsilon` (ε):
    - `init`: 1.0
    - `end`: 0.04
    - Linear decay over `exploration_fraction` = 0.16 (the first 16% of `total_steps`)

**Note**: DQN, like many Deep RL algorithms, is very sensitive to hyperparameters. You can try tuning them, but if they are not set appropriately, the model might perform randomly or achieve very low rewards during testing! Be careful when tuning `total_steps`, `buffer_size`, `gamma`, `learning_rate`, `train_frequency`, `epochs`, and `ε`!

## Results

Below are the results from training with and without soft updates.

<p float="left">
  <img src="figure\soft update.png" alt="soft update" width="500" height="300"/>
  <img src="figure\no soft update.png" alt="no soft update" width="500" height="300"/>
</p>

The trained model successfully achieved the maximum total reward:
- In both cases (with and without soft updates), the model quickly reaches the maximum total reward (500) during testing and maintains this level afterward.
- The average reward during training fluctuates between 200-300 and shows no clear upward trend. This is not critical, as training involves exploration (randomness).
- This environment is very simple, primarily serving to ensure that the code works, so a performance comparison between the two update methods is not conclusive.

**Notes**:
- The code may contain some bugs.
- A chatbot was used to correct typos and format the code in this project.

## References
- [DQN Paper](https://arxiv.org/pdf/1312.5602)
- [rl-baselines3-zoo Hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)