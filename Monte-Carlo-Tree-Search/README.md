# Monte Carlo Tree Search

## Introduction

Monte Carlo Tree Search (MCTS) is an RL algorithm that combines Monte Carlo (performing random actions and collecting statistics) with tree-based search techniques to perform heuristic search for action selection. MCTS is a famous algorithm and serves as the foundation for advanced RL algorithms like AlphaGo, AlphaZero, MuZero, etc.

## Algorithm

Starting with state $s_0$, the algorithm will repeatedly use MCTS to select the best action $a_t$ for state $s_t$ with $t \in 0..T$, where T is the terminal state. The best action $a_t$ will be executed and receive reward $r_t$ and next state $s_{t+1}$.

MCTS is divided into the following components:
- **node**: A node on the tree will include:
    - N: number of visits (number of times passed through during exploration).
    - value: total return (total reward) of episodes from explorations through that node (for easier coding, we store the total instead of calculating expected return).
    - Need to store the environment starting from this node's state (using copy.deepcopy) for easy simulation.
- **explore**: 
    - From the tree root, select a leaf node (node that needs further exploration, node to simulate).
        - From root, recursively use the score function to select the child node with the highest score, stop when reaching a terminal node or an unexplored node.
        - If stopping at an unexplored node: randomly select an unexplored action (no child node for this action yet), create a new child node, move down to this child node.
        - Simulate a random episode with this node.
        - Backpropagate to the root node.
- **Score function**: 
    - Function calculated according to the formula
        - new node (N=0): $score = \inf$
        - explored node (N>0): $score = \frac{V}{N} + c \sqrt\frac{2\ln(N_p)}{N}$
    - With $c = \frac{1}{\sqrt 2}$, $N_p$ is the number of visits of the parent node if this node has a parent (not root) or itself (if it's root).
    - c is recommended to be $\sqrt 2$ or $\frac{1}{\sqrt 2}$, increasing c will reduce dependence on the value (expected return) currently estimated for that node (encouraging more exploration), decreasing c will make the exploration process trust (depend on) the currently estimated value (expected return).
    - Some references uses $score = \frac{V}{N} + c \sqrt\frac{\ln(N_p)}{N}$ with $c = \sqrt 2$
- **Simulation**: perform random actions starting from the node's state until terminal state.
- **Backpropagation**: update the value (expected return) of nodes by adding the return from the just-simulated episode and increment the visit count by 1 for all nodes traversed during the exploration process.
- **select best action**: return the action with the highest visit count during the search process (if there are multiple best actions --> select randomly).

## Code Structure

Code for testing MCTS with **CartPole-v1** in [this notebook](MCTS.ipynb). Note: the code may contain some bugs or may not be optimal!!!
[nodebook v2](MCTS_v2.ipynb) is clean version suitable with AlphaZero and Muzero pseudo codes

Need to tune the following hyperparameters:
- TOTAL_EPISODE: number of episodes to test (since MCTS is random, each episode will give different results, see **Result** section)
- GAMMA: tune GAMMA from 0.9 to 1
- REUSE_TREE: there are 2 variants of MCTS:
    - REUSE_TREE = FALSE: rebuild tree from scratch for each $s_t$.
    - REUSE_TREE = True: reuse the subtree already built from $s_{t-1}$ for $s_t$, just need to delete (detach parent for this child node).
- TIMEOUT: number of seconds in one search:
    - NULL if no time limit desired
    - need to set greater than 0 if SEARCH_STEP = 0 or SEARCH_STEP = NULL
- SEARCH_STEP: number of exploration steps in one search:
    - NULL if wanting to limit time (use TIMEOUT)
    - need to set greater than 0 if SEARCH_STEP = 0 or SEARCH_STEP = NULL

## Result

Results were tested with $gamma \in [0.9, 0.99, 0.997]$, SEARCH_STEP $\in [10, 20, 50, 100, 200, 500]$. With gamma = 0.997, I also tried SEARCH_STEP = 1000 and REUSE_TREE = True.

Result table when tuning gamma and SEARCH_STEP.

<div align="center">

Results

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

With gamma = 0.997:
- I tried increasing SEARCH_STEP = 1000:
    - Average total rewards is **$383.6 \pm 104.79$**
    - But requires an average of **21 minutes 58 seconds** to complete. 
    - Combined with the result table above, we can see that total rewards don't increase when increasing SEARCH_STEP (or at least need to increase SEARCH_STEP very high, increasing by several times won't have clear effects) but the runtime is very long --> not worth trying more.
- Testing with REUSE_TREE:
    - Results in the table below 
    - When using REUSE TREE, the number of searches can be reduced while still giving good results (work with only 10 searchs). However, when the number of searches is increased, the results will not increase (or decrease).
    - With the above results, using REUSE TREE doesn't show significant effectiveness (even worse) while search time is very long --> not worth trying more.

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

**Notes**:
- Runtime is relative as it depends on hardware!
- Since std is quite high, might need to run more than 50 or 100 times to be more objective.
- Project uses Chatbot to correct spelling or format code!

## Reference
- [geeksforgeeks MCTS](https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/)
- [medium _michelangelo_ MCTS for dummies](https://medium.com/@_michelangelo_/monte-carlo-tree-search-mcts-algorithm-for-dummies-74b2bae53bfa)
- [gibberblot mcts](https://gibberblot.github.io/rl-notes/single-agent/mcts.html)