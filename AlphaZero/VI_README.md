# AlphaZero

## Introduction

AlphaZero là một thuật toán trí tuệ nhân tạo tiên tiến do DeepMind (thuộc Google) phát triển, nổi bật với khả năng tự học để đạt trình độ siêu việt trong các trò chơi chiến lược như cờ vua, cờ vây (Go) và shogi (cờ Nhật). AlphaZero cũng có thể được huấn luyện để giải quyết các các game hoặc mô phỏng đơn giản khác. 

AlphaZero là cải tiến từ AlphaGo, có thể tự học bằng self play (tương tự các thuật toán Reinforcement khác) mà không cần dữ liệu cách chơi thu thập từ người chơi như AlphaGo.

Ý tưởng chính của AlphaZero là kết hợp mạng neural học sâu và Monte Carlo Tree Search (MCTS). Thay thế việc giả lập 1 lần chơi ngẫu nhiên (random simulate) bằng kết quả từ neural network. Lặp lại việc sử dụng MCTS và mạng neural đang huấn luyện để thu thập dữ liệu và dùng dữ liệu này huấn luyện mô hình.

## Algorithm

## Code Structure

Project sử dụng jupyter nodebook, test trên environment **CartPole-v1**:
- Notebook AlphaZero [AlphaZero.ipynb](AlphaZero.ipynb)
- Notebook AlphaZero chạy song song nhiều worker (hoặc actor) để thu thập dữ liệu huấn luyện và testing [AlphaZero_multi_worker.ipynb](AlphaZero_multi_worker.ipynb)

## Result

Dưới đây là kết quả 2 lần chạy khi dùng loss là Scale Cross Entropy (CE):

<p float="left">
  <img src="figure\CE1.png" alt="Kết quả CE lần 1" width="500" height="300"/>
  <img src="figure\CE2.png" alt="Kết quả CE lần 2" width="500" height="300"/>
</p>

Dưới đây là kết quả 2 lần chạy khi dùng loss là MSE:

<p float="left">
<img src="figure\MSE1.png" alt="Kết quả MSE lần 1" width="500" height="300"/>
  <img src="figure\MSE2.png" alt="Kết quả MSE lần 2" width="500" height="300"/>
</p>

Có thể thấy train với CE giúp model học nhanh hơn và đảm bảo đạt tổng phần thưởng (return hay total rewards) tối đa trong cả 2 lần chạy. MSE học chậm hơn và chưa đạt được 500 rewards trong lần chạy thứ 2 (có thể cần thêm episode hoặc sẽ không hội tụ?).

Khi dùng CE, thuật toán giúp model học rất nhanh và đạt tổng phần thưởng tối đa. Trong quá trình huấn luyện, model có thể duy trì phần thưởng tối đa 500 trong nhiều episode, tốt hơn MCTS cơ bản vì trong quá trình search ngẫu nhiên, đôi khi quá trình mở rộng sẽ không xây dựng được cây đủ tốt --> MCTS dựa vào cây và random sẽ cho kết quả tệ vì cây không tốt, AlphaZero được model bù đắp nên kết quả vẫn tốt hơn và cây cũng phụ thuộc nhiều vào model nên cây được xây dựng cũng tốt hơn cây được random và similate (simulate vẫn dùng random!) nhiều hơn của MCTS thông thường.

**Lưu ý**:
- Project có dùng Chatbot để chỉnh lỗi chính tả hoặc format code!

## Reference
- [medium _michelangelo_ MCTS for dummies](https://medium.com/@_michelangelo_/alphazero-for-dummies-5bcc713fc9c6)
- [erenon AlphaZero pseudo code](https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023)