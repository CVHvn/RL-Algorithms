# Deep-Q-Network-variants

## Deep Q Network (DQN)
`Deep Q-Learning` (`Deep Q-Network`) is a `Reinforcement Learning` (`RL`) algorithm that combines the `classic Q-Learning` algorithm and `Deep Neural Networks`. This is my minimal Pytorch implementation of [Deep Q Network (DQN)](DQN). Refer to the original paper [Deep Q-Network (DQN)](https://arxiv.org/pdf/1312.5602).

## Double Deep Q Network (Double DQN)
`Double Deep Q-Learning` is an improvement on the `Deep Q-Learning` algorithm, taking inspiration from [classic Double Q learning](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf) to reduce the overestimation bias that `DQN` encounters. This is my minimal Pytorch implementation of [Double Deep Q Network (Double DQN)](DoubleDQN). Refer to the original paper [Double Deep Q-Network (Double_DQN)](https://arxiv.org/pdf/1509.06461).

## Dueling Deep Q Network (Dueling DQN)
`Dueling DQN` is an improvement on the `Deep Q-Learning` algorithm, applying the Dueling architecture: splitting $Q(s, a)$ into $V(s)$ and $A(s, a)$. This helps the Model train faster and better distinguish the value of the state $V(s)$ from the value of each action $A(s, a)$ corresponding to that state. Refer to the original paper [Dueling Deep Q-Network (Dueling DQN)](https://arxiv.org/pdf/1511.06581)

## Dueling Double Deep Q Network (D3QN)
`D3QN` is a combination of the Dueling architecture from Dueling DQN: splitting Q into V and A, and the Target Network calculation method from Double DQN: using the Online Model to select the next action $a'$ and using the Target Model to calculate $Q(s', a')$.

## Deep Recurrent Q-Learning (DRQN)
`DRQN` is a combination of DQN and RNN. There are several variations of DRQN, but this project only implements two versions: the original DRQN from the [DRQN paper](https://arxiv.org/pdf/1507.06527) and Double DRQN, which is part of the [ViZDoom paper](https://arxiv.org/pdf/1801.01000).

## Reference
- [DQN paper](https://arxiv.org/pdf/1312.5602)
- [Double DQN paper](https://arxiv.org/pdf/1509.06461)
- [Double Q-learning paper](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
- [Dueling DQN paper](https://arxiv.org/pdf/1511.06581)
- [DRQN paper](https://arxiv.org/pdf/1507.06527)
- [ViZDoom paper](https://arxiv.org/pdf/1801.01000)
- [rl-baselines3-zoo hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)