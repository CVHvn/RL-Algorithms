# Algorithm

**AlphaZero là nâng cấp của MCTS. Các ideas chính:**
- AlphaZero thêm vào 2 mạng neural để học **value** và **policy** từ 1 state input. 
- Khi expand 1 node mới, thay vì similate bằng cách chơi ngẫu nhiên như MCTS, AlphaZero sẽ dùng **value network** để dự đoán expected return của node cần expand. 
- **Policy network** cũng được sử dụng để cải tiến hàm score của các node và khuyến khích MCTS khám phá tốt hơn.

**AlphaZero có thể chia thành 2 phần:**
- Dùng MCTS để thu thập dữ liệu (các episodes). Sử dụng multi worker để collect dữ liệu nhanh hơn.
- Sử dụng dữ liệu để huấn luyện mô hình.
- Lặp lại quá trình trên cho đến khi model hội tựu hoặc hết thời gian training.

### MCTS của AlphaZero

Khởi đầu với state $s_0$, thuật toán sẽ lặp lại việc dùng MCST để chọn action $a_t$ cho state $s_t$ với $t \in 0..T$, T là terminal state. action $a_t$ sẽ được thực hiện và nhận về phần thưởng $r_t$ và next state $s_{t+1}$ (khác với MCTS cơ bản chọn best action dựa trên visited count của các node con, AlphaZero sẽ sample action theo xác suất là visited count của các node con để khuyến khích khám phá). 

MCST được chia nhỏ thành các phần sau:
- **node**: 1 node trên cây sẽ gồm:
    - N: số lần visit (số lần được đi qua khi explore).
    - value: tổng return (tổng phần thưởng) các episode của các lần explore qua node đó (để dễ code sẽ lưu tổng thay vì tính expected return).
    - prior: xác suất node cha đi đến node này (dự đoán bằng policy model lúc tạo các nút con).
    - Cần lưu environment bắt đầu từ state của node này (sử dụng copy.deepcopy) để dễ dàng simulate.
- **add_exploration_noise**: thêm noise vào các node của của gốc dựa vào prior của các node này để khuyến khích khám phá (sẽ không thêm khi test model).
- **explore**: 
    - Từ gốc cây (root), chọn ra nút lá (node cần khám phá thêm, nút cần giả lập).
        - Từ root, đệ quy quá trình dùng hàm score để chọn nút con có score cao nhất, dừng lại khi đến terminal node hoặc 1 node chưa khám phá hết.
        - Nếu dùng lại ở node chưa được khám phá hết: tạo tất cả node con cho node này, dùng policy model để tính prior cho các node con.
        - Giả lập (simulate) 1 episode bằng value model.
        - Backpropagation về nút gốc.
- **Hàm score**: 
    - Hàm được tính theo công thức
        - node mới (N=0): $score = \inf$
        - node đã explore (N>0): 
            - $pb_c = (\ln\frac{N_p + pb_{c base} + 1}{pb_{c base}} + pc_{c init}) \sqrt\frac{N_p}{N+1}$
            - $prior score = prior pb_c$
            - $score = min max scale(\frac{V}{N}) + prior score$
    - Với $pb_{c base} = 19652$, $pc_{c init} = 1.25$, $N_p$ là số lần visit của nút cha, $minmaxscale$ là hàm scale value của node dựa theo min max value của các node trong 1 lần MCTS (sẽ scale từ lần update min max scale thứ 2 vì lần đầu chỉ có 1 giá trị không cần scale dựa theo tất cả tài liệu tìm được).
    - Các giá trị pb_c được giữ nguyên trong nhiều paper liên quan nên không cần tuning!
- **Giả lập**: Sử dụng value network để tính expected return
- **Backpropagation**: cập nhật value (expected return) của các node thêm return từ episode giả lập vừa thực hiện và số lần visit của node lên 1 cho tất cả các node đã đi qua trong quá trình explore.
- **chọn action**: nếu testing thì trả về hành động có N cao nhất, ngược lại sample action dựa trên xác suất N của các node (**lưu ý** trong paper gốc sẽ tính temperature dựa trên số step đã train, sau đó sample theo temperature_softmax nhưng với environment đơn giản thì không cần thiết nên tôi không triển khai!).

### Huấn luyện:
- Lưu các episode của MCTS vào replay buffer.
- Sample `batchsize` samples từ replay buffer:
    - Cần sample:
        - `state` (để model predict và train)
        - `target value` (target để train value network)
        - `target policy` (target để train policy network)
    - Với thuật toán AlphaZero, có thể lưu data dựa theo `state` như DQL mà không cần lưu theo episode!
    - Có thể dùng priority cho các episode dựa theo độ dài episode (không được tôi triển khai)!
- train value network:
    - `target value` là value của node ứng với `state` đó dựa theo MCTS. Có thể dùng N-step value dựa theo episode như Muzero (không được tôi triển khai).
    - Có thể sử dụng loss là MSE để đơn giản hóa.
    - Các paper và nghiên cứu liên quan (bao gồm project này) khuyến nghị dùng scaler cross entropy loss để huấn luyện:
        - Sử dụng 1 support vector $Z = z_0 \.. z_n$ có n+1 phần tử căn đều nhau (ví dụ 0, 5, 10, 15, ..., 50).
        - value network sẽ dự đoán xác suất value V thuộc $z_i$ (gọi là $pz_i$).
        - Khi đó $V = \sum_{i=0}^{n} {z_i pz_i}$ 
        - Với số thực V', có thể chuyển V' về phân phối sác suất dựa trên Z.
        - Khi đó sẽ dùng Cross entropy để train xác suất model dựa đoán với suất suất khi convert target value dựa trên Z.
        - **lưu ý**: có nhiều cách triển khai scaler cross entropy, đây chỉ là cách đơn giản được tôi triển khai!
- train Policy network:
    - `target policy` là xác suất dựa trên số lần visited của node ứng với `state` đó trong MCTS.
    - Huấn luyện bằng Cross entropy!