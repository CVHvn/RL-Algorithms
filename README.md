# RL-Algorithms
Reinforcement learning algorithms

# Introduction & Motivation
This project is minimal implementation of some RL algorithms. When I read new RL algorithms (especially after a long time without coding RL), I can't remember basic RL algorithms and take a lot of time to remember old algorithms. And many RL sources (Github, blog, paper) use different styles and variants of RL algorithms, which makes it hard to read. I implement some RL algorithms to remember and reuse my code when I want to learn and try something news.

# Algorithms

Each folder includes a README file (containing the algorithm description, testing results, notes, and important hyperparameters).

- [Basic Algorithm](Basic_Algorithms):
    - [Monte-Carlo](Basic_Algorithms\Monte-Carlo)
    - [SARSA](Basic_Algorithms\SARSA)
    - [Q-Learning](Basic_Algorithms\Q-Learning)
    - [Double-Q-Learning](Basic_Algorithms\Double-Q-Learning)
- [Deep Q Learning](Deep-Q-Network-variants): 
    - [Deep-Q-Network (DQN)](Deep-Q-Network-variants/DQN)
    - [Double Deep-Q-Network (DQN)](Deep-Q-Network-variants/DoubleDQN)
    - [Dueling Deep-Q-Network (DQN)](Deep-Q-Network-variants/Dueling_DQN)
    - [Dueling Double Deep-Q-Network D3QN](Deep-Q-Network-variants/D3QN)
    - [Deep Recurrent Q-Learning (DRQN)](Deep-Q-Network-variants/DRQN):
      - [Double Deep Recurrent Q-Learning (Double-DRQN)](Deep-Q-Network-variants/DRQN) 
- [Policy-Based Method](Policy_based_method):
    - [Policy Gradient](Policy_based_method/Policy_Gradient):
        - [REINFORCE](Policy_based_method/Policy_Gradient/REINFORCE)
        - [REINFORCE with baseline](Policy_based_method/Policy_Gradient/REINFORCE_with_baseline)
- [Monte Carlo Search Tree](Monte-Carlo-Tree-Search)
- [AlphaZero](AlphaZero)

# TODO

- Clean code (some of the algorithms I coded are quite messy; I will clean them up later).
- Run additional experiments with more complex environment.

# Reference