# policy based method
Different from value-based algorithms (the goal is to build a method to calculate the value function of a state V(s) or (state, action) Q(s, a)) such as Q-Learning, SARSA, ... policy based algorithms will focus on building a policy Neural Network to directly determine the action from the state instead of through value function types like value based.

## [policy gradient](Policy_Gradient)
Policy gradient is a popular branch in RL, based on the policy gradient theorem. there are many policy gradient algorithms:
- [REINFORCE](Policy_Gradient\REINFORCE) also known as Monte Carlo Policy Gradient.
- [REINFORCE with baseline](Policy_Gradient\REINFORCE_with_baseline): adds a baseline to reduce variance.
- Actor Critic and advanced variants of Actor Critic such as A2C, A3C, ACTKR, PPO, ACER, ...