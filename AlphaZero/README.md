# AlphaZero

## Introduction

This project is my minimal implementation of [AlphaZero](https://arxiv.org/pdf/1712.01815).

`AlphaZero` is an advanced artificial intelligence algorithm developed by DeepMind (Google), distinguished by its ability to self-learn to achieve superhuman performance in strategic games such as chess, Go, and shogi (Japanese chess). AlphaZero can also be trained to solve other simple games or simulations.

AlphaZero is an improvement upon AlphaGo, capable of self-learning through self-play (similar to other Reinforcement Learning algorithms) without needing game data collected from human players (AlphaGo). AlphaZero is generalized the algorithm of AlphaGo Zero for many environments other than Go (AlphaGo --> AlphaGo Zero --> AlphaZero). According to research, AlphaGo will imitate the way humans play, while AlphaZero can outperform humans by finding different ways of acting.

The core idea of AlphaZero is to combine deep neural networks and Monte Carlo Tree Search (MCTS). It replaces random game simulations with results from a neural network. This process of using MCTS and the training neural network is iterated to collect data, which is then used to train the model.

## Algorithm

AlphaZero is an upgrade to MCTS. The main ideas are:
- AlphaZero adds two neural networks to learn value and policy from a state input.
- When expanding a new node, instead of simulating with random play as in MCTS, AlphaZero uses the value network to predict the expected return of the node to be expanded.
- The policy network is also used to improve the score function of the nodes and encourage MCTS to explore better.

AlphaZero can be divided into two parts:
- Using MCTS to collect data (episodes). Using multi-workers to collect data faster.
- Using the collected data to train the model.
- Repeat the above process until the model converges or training time runs out.

To speed up training (like the original paper), multiple workers (actors) are needed to collect data. I use `ray` because it's easy to code (you can use `multiprocess` to speed up if you're good at multiprocessing, I encountered quite many issues using `multiprocess` with GPUs, so I had to use `ray` :3).

[Detail Algorithm](Algorithm_README.md)

## Code Structure

The project uses Jupyter notebooks and is tested on the `**CartPole-v1**` environment:
- AlphaZero Notebook [AlphaZero.ipynb](AlphaZero.ipynb)
- AlphaZero Notebook running multiple workers (or actors) in parallel to collect training data and for testing [AlphaZero_multi_worker.ipynb](AlphaZero_multi_worker.ipynb)

Note: the code may contain some bugs or not be fully optimized!!!

## Trained Model

You can load [trained model](trained_model)

## Hyperparameter

The following hyperparameters need to be tuned:
- `num_workers`: number of workers (actors) running in parallel. Each worker will run 1 episode using MCTS.
- `gamma`: tuning from 0.9 to 1.
- `total_episode`: total number of training episodes.
- `search_step`: number of exploration steps in one search (default is 50; according to the documentation I've read, increasing it is better but takes a very long time to run).
- `start_train_from_episode`: start training from episode `start_train_from_episode`. For simple environments, I found this parameter doesn't have a significant impact.
- `buffer_size`: number of episodes stored in the replay buffer. Increase depending on the environment's difficulty (increasing too much will consume memory!).
- `epochs`: number of epochs in one training run.
- `training_steps`: train every `training_steps` episodes.
- `learning_rate`
- `batchsize`
- `testing_steps`: test every `testing_steps` episodes.
- `total_test_episode`: number of episodes per test run. Increase this to ensure the model performs well (because MCTS runs randomly, testing with only a few episodes doesn't guarantee objectivity; the model might be poor but achieve good results in a random episode due to good search!).
- Other parameters should be kept at their default values to ensure the algorithm runs correctly!

## Result

Below are the results of two runs when using Scale Cross Entropy (CE) as the loss function:

<p float="left">
  <img src="figure\CE1.png" alt="CE first run result" width="500" height="300"/>
  <img src="figure\CE2.png" alt="CE second run result" width="500" height="300"/>
</p>

Below are the results of two runs when using MSE as the loss function:

<p float="left">
<img src="figure\MSE1.png" alt="MSE first run result" width="500" height="300"/>
  <img src="figure\MSE2.png" alt="MSE second run result" width="500" height="300"/>
</p>

It can be observed that training with CE helps the model learn faster and ensures reaching the maximum total reward (return or total rewards) in both runs. MSE learns slower and did not achieve 500 rewards in the second run (it might need more episodes or might not converge?).

When using CE, the algorithm enables the model to learn very quickly and reach the maximum total reward. During training, the model can maintain the maximum reward of 500 for many episodes, which is better than basic MCTS because in a random search, sometimes the expansion process might not build a good enough tree --> MCTS, relying on the tree and randomness, will yield poor results if the tree is not good. AlphaZero is compensated by the model, so the results are still better, and the tree built is also much better than a randomly generated and simulated tree (simulation still uses randomness!) of a regular MCTS.

**Notes**: Project uses Chatbot to correct spelling or format code!

## Reference
- [medium _michelangelo_ MCTS for dummies](https://medium.com/@_michelangelo_/alphazero-for-dummies-5bcc713fc9c6)
- [erenon AlphaZero pseudo code](https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023)
- [AlphaZero paper](https://arxiv.org/pdf/1712.01815)