# Monte Carlo Tree Search

## Introduction

Monte Carlo Tree Search (MCTS) là 1 thuật toán RL kết hợp Monte Carlo (thực hiện hành động ngẫu nhiên và thống kê kết quả) và kỹ thuật tìm kiếm dựa trên cây để thực hiện heuristic search chọn action. MCTS là một thuật toán nổi tiếng và là nền tảng của các thuật toán RL nâng cao như AlphaGo, AlphaZero, Muzero, ... 

## Algorithm

Khởi đầu với state $s_0$, thuật toán sẽ lặp lại việc dùng MCST để chọn best action $a_t$ cho state $s_t$ với $t \in 0..T$, T là terminal state. Best action $a_t$ sẽ được thực hiện và nhận về phần thưởng $r_t$ và next state $s_{t+1}$.

MCST được chia nhỏ thành các phần sau:
- **node**: 1 node trên cây sẽ gồm:
    - N: số lần visit (số lần được đi qua khi explore).
    - value: tổng return (tổng phần thưởng) các episode của các lần explore qua node đó (để dễ code sẽ lưu tổng thay vì tính expected return).
    - Cần lưu environment bắt đầu từ state của node này (sử dụng copy.deepcopy) để dễ dàng simulate.
- **explore**: 
    - Từ gốc cây (root), chọn ra nút lá (node cần khám phá thêm, nút cần giả lập).
        - Từ root, đệ quy quá trình dùng hàm score để chọn nút con có score cao nhất, dừng lại khi đến terminal node hoặc 1 node chưa khám phá hết.
        - Nếu dùng lại ở node chưa được khám phá hết: chọn ngẫu nhiên 1 hành động chưa được khám phá (chưa có node con này), tạo node con mới, đi xuống node con này.
        - Giả lập (simulate) 1 episode ngẫu nhiên với node này.
        - Backpropagation về nút gốc.
- **Hàm score**: 
    - Hàm được tính theo công thức
        - node mới (N=0): $score = \inf$
        - node đã explore (N>0): $score = \frac{V}{N} + c \sqrt\frac{2\ln(N_p)}{N}$
    - Với $c = \frac{1}{\sqrt 2}$, $N_p$ là số lần visit của nút cha nếu node này có cha (không phải root) hoặc chính nó (nếu là root).
    - c được khuyến nghị là $\sqrt 2$ hoặc $\frac{1}{\sqrt 2}$, tăng c sẽ làm giảm sự phụ thuộc vào value (expected return) đang ước tính cho node đó (khuyến khích khám phá thêm), giảm c sẽ làm quá trình explore tin tưởng (phụ thuộc) và value (expected return) đang ước tính.
    - Một số tài liệu sử dụng $score = \frac{V}{N} + c \sqrt\frac{\ln(N_p)}{N}$ với $c = \sqrt 2$
- **Giả lập**: thực hiện hành động ngẫu nhiên bắt đầu từ state của node đến terminal state.
- **Backpropagation**: cập nhật value (expected return) của các node thêm return từ episode giả lập vừa thực hiện và số lần visit của node lên 1 cho tất cả các node đã đi qua trong quá trình explore.
- **chọn best action**: trả về hành động có số lượt visit cao nhất trong quá trình search (nếu có nhiều hành động tốt nhất --> chọn ngẫu nhiên).

## Code Structure

Code chạy thử nghiệm MCTS với **CartPole-v1** trong [nodebook này](Monte-Carlo-Tree-Search\MCTS.ipynb). Lưu ý: code có thể tồn tại một số bug hoặc không tối ưu!!!
[nodebook v2](Monte-Carlo-Tree-Search\MCTS_v2.ipynb) được code clean để giống với pseudo code của AlphaZero và Muzero

Cần tunning các siêu tham số sau:
- TOTAL_EPISODE: số episode sẽ test (vì MCTS ngẫu nhiên nên mỗi episode sẽ cho kết quả khác nhau, xem phần **Result**)
- GAMMA: tinh chỉnh GAMMA từ 0.9 đến 1
- REUSE_TREE: có 2 biến thể của MCTS:
    - REUSE_TREE = FALSE: built lại tree từ đầu với mỗi $s_t$.
    - REUSE_TREE = True: tái sử dụng lại cây con đã build từ $s_{t-1}$ cho $s_t$, chỉ cần xóa (detach parent cho nút con này).
- TIMEOUT: số giây trong 1 lần search:
    - NULL nếu không muốn giới hạn thời gian
    - cần gán lớn hơn 0 nếu SEARCH_STEP = 0 hoặc SEARCH_STEP = NULL
- SEARCH_STEP: số step explore trong 1 lần search:
    - NULL nếu muốn giới hạn thời gian (dùng TIMEOUT)
    - cần gán lớn hơn 0 nếu SEARCH_STEP = 0 hoặc SEARCH_STEP = NULL

## Result

Kết quả được thử nghiệm với $gamma \in [0.9, 0.99, 0.997]$, SEARCH_STEP $\in [10, 20, 50, 100, 200, 500]$. Với gamma = 0.997, tôi có thử thêm với SEARCH_STEP = 1000 và REUSE_TREE = True.

Bảng kết quả khi tunning gamma và SEARCH_STEP.

<div align="center">

Kết quả

| Steps  | Mean ± Std (γ=0.99) | Mean ± Std (γ=0.997) | Mean ± Std (γ=1)    | Time (mean)  |
|--------|----------------------|---------------------|---------------------|--------------|
| 10     | 402.9 ± 104.68       | **419.0 ± 107.51**  | 376.4 ± 109.66      | ~0:04        |
| 20     | 389.8 ± 119.12       | **395.8 ± 101.98**  | 387.1 ± 118.68      | ~0:08        |
| 50     | **424.4 ± 106.54**   | 421.3 ± 78.81       | 374.9 ± 82.62       | ~0:23        |
| 100    | 426.3 ± 85.60        | **450.7 ± 82.47**   | 347.3 ± 113.70      | ~0:53        |
| 200    | **391.8 ± 136.19**   | 322.8 ± 88.25       | 390.4 ± 83.91       | ~2:00        |
| 500    | 396.0 ± 109.84       | 368.3 ± 78.95       | **403.9 ± 91.12**   | ~7:36        |
| 1000   | —                    | 383.6 ± 104.79      | —                   | 21:58        |

</div>

Với gamma = 0.997:
- Tôi thử tăng SEARCH_STEP = 1000:
    - Tổng rewards trung bình là **$383.6 \pm 104.79$**
    - Nhưng cần tới trung bình **21 phút 58 giây** để chạy xong. 
    - Kết hợp với bảng kết quả trên, có thể thấy tổng rewards không được tăng khi tăng SEARCH_STEP (hoặc ít nhất cần tăng SEARCH_STEP lên rất cao, tăng lên gấp vài lần sẽ không có tác dụng rõ ràng) nhưng thời gian chạy rất lâu --> không đáng thử thêm.
- Thử với REUSE_TREE:
    - Kết quả ở bảng bên dưới
    - Khi dùng REUSE TREE có thể giảm số lần search mà vẫn cho kết quả tốt (chỉ cần search 10 lần). Tuy nhiên khi tăng số lần search thì kết quả không tăng thêm (hoặc giảm đi).
    - Với kết quả trên thì việc dùng REUSE TREE sẽ không cho thấy hiệu quả đáng kể (thậm chí tệ hơn) mà thời gian search rất lâu --> không đáng thử thêm.

<div align="center">

REUSE_TREE gamma = 0.997

| Steps | Mean ± Std       | Time     |
|-------|------------------|----------|
| 10    | 467.1 ± 60.01    | 00:19  |
| 20    | 326.4 ± 100.66   | 00:21  |
| 50    | 343.3 ± 79.01    | 01:03  |
| 100   | 400.9 ± 101.57   | 02:49  |
| 200   | 437.6 ± 97.10    | 07:10  |
| 500   | 387.5 ± 81.05    | 15:34  |

</div>

**Lưu ý**:
- Thời gian chạy mang tính tương đối vì phụ thuộc phần cứng!.
- Vì std khá cao nên có thể cần chạy nhiều hơn 50 hoặc 100 lần để khách quan hơn.
- Project có dùng Chatbot để chỉnh lỗi chính tả hoặc format code!

## Reference
- [geeksforgeeks MCTS](https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/)
- [medium _michelangelo_ MCTS for dummies](https://medium.com/@_michelangelo_/monte-carlo-tree-search-mcts-algorithm-for-dummies-74b2bae53bfa)
- [gibberblot mcts](https://gibberblot.github.io/rl-notes/single-agent/mcts.html)