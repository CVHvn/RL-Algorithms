# Deep-Q-Network-variants

## Deep Q Network (DQN)
`Deep Q-Learning` (`Deep Q-Network`) là 1 thuật toán Reinforcement learning (RL) kết hợp thuật toán `Q-Learning cổ điển` và `Deep Neural Network`. Đây là Pytorch minimal implementation [Deep Q Network (DQN)](DQN) của tôi. Tham khảo paper gốc [Deep Q-Network (DQN)](https://arxiv.org/pdf/1312.5602)

## Double Deep Q Network (Double DQN)
`Double Deep Q-Learning` là cải tiến của thuật toán `Deep Q-Learning`, lấy ý tưởng từ [Double Q learning cổ điển](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf) nhằm giảm overestimation bias mà `DQN` gặp phải. Đây là Pytorch minimal implementation [Double Deep Q Network (Double DQN)](DoubleDQN) của tôi. Tham khảo paper gốc [Double Deep Q-Network (Double DQN)](https://arxiv.org/pdf/1509.06461)

## Dueling Deep Q Network (Dueling DQN)
`Dueling DQN` là cải tiến của thuật toán `Deep Q-Learning`, áp dụng kiến trúc Dueling: tách $Q(s, a)$ thành $V(s)$ và $A(s, a)$ giúp Model train nhanh hơn và phân biệt tốt hơn giá trị của state $V(s)$ và giá trị của từng action $A(s, a)$ ứng với state đó. Tham khảo paper gốc [Dueling Deep Q-Network (Dueling DQN)](https://arxiv.org/pdf/1511.06581)

## Dueling Double Deep Q Network (D3QN)
`D3QN` là sự kết hợp giữa kiến trúc Dueling trong Dueling DQN: tách Q thành V và A và cách tính Target Network trong Double DQN: dùng Online Model để chọn next action $a'$ và dùng Target Model để tính $Q(s', a')$.

## Deep Recurrent Q-Learning (DRQN)
`DRQN` là sự kết hợp DQN và RNN. Có nhiều phiên bản khác nhau của DRQN, project này chỉ implement 2 version DRQN gốc trong paper [DRQN paper](https://arxiv.org/pdf/1507.06527) và Double DRQN, 1 phần của [ViZDoom paper](https://arxiv.org/pdf/1801.01000).

## Reference
- [DQN paper](https://arxiv.org/pdf/1312.5602)
- [Double DQN paper](https://arxiv.org/pdf/1509.06461)
- [Double Q-learning paper](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
- [Dueling DQN paper](https://arxiv.org/pdf/1511.06581)
- [DRQN paper](https://arxiv.org/pdf/1507.06527)
- [ViZDoom paper](https://arxiv.org/pdf/1801.01000)