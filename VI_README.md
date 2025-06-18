# RL-Algorithms
Reinforcement learning algorithms

# Introduction & Motivation
Project bao gồm minimal implementation của 1 số thuật toán RL. Khi tôi đọc các thuật toán RL mới (đặc biệt là sau 1 thời gian dài không code về RL), tôi không thể nhớ các thuật toán cơ bản và sẽ mất nhiều thời gian để nhớ lại các thuật toán này. Các nguồn tham khảo (Github, blog, paper) cũng dùng các biến thể khác nhau và triển khai với phong cách khác nhau làm tôi khó nhớ lại hơn. Tôi triển khai các thuật toán này để dễ nhớ và tận dụng lại khi tìm hiểu và thử nghiệm các project RL mới.

# Algorithms

Mỗi folder sẽ gồm file README (mô tả sơ qua về thuật toán, kết quả chạy thử và 1 số lưu ý, siêu tham số quan trọng) và file notebook.

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
- [Monte Carlo Search Tree](Monte-Carlo-Tree-Search)
- [AlphaZero](AlphaZero)

# TODO

- Clean code (một số thuật toán tôi code rất xấu, tôi sẽ clean lại sau).
- Chạy thêm thử nghiệm với env phức tạp hơn.

# Reference