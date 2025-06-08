# Algorithm

**AlphaZero is an upgrade of MCTS. The main ideas are:**
- AlphaZero adds two neural networks to learn the **value** and **policy** from a given state input.
- When expanding a new node, instead of simulating by playing randomly like in MCTS, AlphaZero uses the **value network** to predict the expected return of the node to be expanded.
- The **policy network** is also used to improve the scoring function of the nodes and encourage better exploration by MCTS.

**AlphaZero can be divided into two parts**:
- Use MCTS to collect data (episodes). Multiple workers are used to collect data more efficiently.
- Use the collected data to train the model.
- Repeat the above process until the model converges or the training time runs out.

### AlphaZero MCTS

Starting with state $s_0$, the algorithm repeatedly uses MCTS to select action $a_t$ for state $s_t$ with $t \in 0..T$, where T is the terminal state. Action $a_t$ will be executed, returning reward $r_t$ and next state $s_{t+1}$. (Different from basic MCTS which selects the best action based on the visited count of child nodes, AlphaZero samples actions according to the probability proportional to the visited counts of the child nodes to encourage exploration.)

MCTS is divided into the following components:
- **Node**: A node on the tree consists of:
    - N: the number of visits (the number of times it has been traversed during exploration).
    - value: the total return (total reward) from episodes that passed through this node during exploration (for easier implementation, the total is stored instead of computing the expected return).
    - prior: the probability of transitioning from the parent node to this node (predicted by the policy model when the child nodes are created).
    - The environment starting from this node’s state should be stored (using copy.deepcopy) to facilitate simulation.
- **add_exploration_noise**: adds noise to the root’s child nodes based on their prior probabilities to encourage exploration (this is not applied when testing the model).
- **explore**:
    - From the root, select a leaf node (node to explore further, node to simulate).
        - From the root, recursively use the score function to select the child node with the highest score, stopping when a terminal node or an unexpanded node is reached.
        - If stopping at an unexpanded node: create all child nodes for this node, use the policy model to calculate the prior for the child nodes.
        - Simulate an episode using the value model.
        - Backpropagation to the root node.
- **Score function**:
    - The function is calculated using the formula:
        - New node (N=0): $score = \inf$
        - Explored node (N>0):
            - $pb_c = (\ln\frac{N_p + pb_{c base} + 1}{pb_{c base}} + pc_{c init}) \sqrt\frac{N_p}{N+1}$
            - $prior score = prior pb_c$
            - $score = min max scale(\frac{V}{N}) + prior score$
    - Where $pb_{c base} = 19652$, $pc_{c init} = 1.25$, $N_p$ is the number of visits of the parent node, $minmaxscale$ is a function to scale the node's value based on the min/max values of nodes in one MCTS run (scaling will occur from the second update of $min max scale$ onwards because the first time there's only one value, so no scaling is needed based on all found documentation).
    - The $pb_c$ values are kept constant across many related papers, so no tuning is required!
- **Simulate**: Use value network to predict expected return.
- **Backpropagation**: update the value (expected return) of the nodes by adding the return from the simulated episode just performed, and increment the visit count (N) by 1 for all nodes traversed during the exploration process.
- **Select action**: If testing, return the action with the highest N. Otherwise, sample an action based on the probability N of the nodes (**note**: in the original paper, temperature is calculated based on the number of training steps, then sampled according to temperature_softmax, but for a simple environment, this is not necessary, so I haven't implemented it!).

### Training:
- Save MCTS episodes to a replay buffer.
- Sample `batchsize` samples from the replay buffer:
    - Need to sample:
        - `state` (for model prediction and training)
        - `target value` (target for training the value network)
        - `target policy` (target for training the policy network)
    - With the AlphaZero algorithm, data can be stored based on `state` like in DQL, without needing to store by episode!
    - Priority can be used for episodes based on episode length (not implemented by me)!
- Train value network:
    - The `target value` is the value of the node corresponding to that `state` based on MCTS. N-step value based on the episode, like MuZero, can be used (not implemented by me).
    - MSE loss can be used for simplification.
    - Related papers and research (including this project) recommend using scaled cross-entropy loss for training:
        - Use a support vector $Z = z_0 \.. z_n$ with n+1 evenly spaced elements (e.g., 0, 5, 10, 15, ..., 50).
        - The value network will predict the probability that value V belongs to $z_i$ (called $pz_i$).
        - Then $V = \sum_{i=0}^{n} {z_i pz_i}$
        - For a real number V', V' can be converted into a probability distribution based on Z.
        - Then Cross-entropy will be used to train the model's predicted probability against the probability when converting the target value based on Z.
        - **note**: there are many ways to implement scaled cross-entropy; this is just a simple way implemented by me!
- Train Policy network:
    - The `target policy` is the probability based on the visit count of the node corresponding to that `state` in MCTS.
    - Train using Cross-entropy!