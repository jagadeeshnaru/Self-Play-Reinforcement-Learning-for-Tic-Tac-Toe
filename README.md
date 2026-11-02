# Tic-Tac-Toe Reinforcement Learning Project

A Python project that trains an agent to play Tic-Tac-Toe using
**tabular reinforcement learning** and then evaluates it against a
**Minimax opponent**.

## Project structure

``` text
tic_tac_toe_rl/
│
├── environment.py   # Tic-Tac-Toe board and game rules
├── agent.py         # Reinforcement learning agent
├── minimax.py       # Minimax opponent
├── training.py      # Training and evaluation functions
├── main.py          # Program entry point
├── requirements.txt
└── README.md
```

## Requirements

-   Python 3.10 or later
-   NumPy

Install the required package:

``` bash
pip install -r requirements.txt
```

Or install NumPy directly:

``` bash
pip install numpy
```

## Run the project

1.  Extract the project ZIP file.
2.  Open the `tic_tac_toe_rl` folder in VS Code or a terminal.
3.  Install the requirements.
4.  Run:

``` bash
python main.py
```

The main program trains the agent for 40,000 episodes against a random
opponent, then evaluates it for 1,000 games against Minimax.

## What each file does

### `environment.py`

Contains the `TicTacToe` class. It manages the board, current player,
legal moves, game state, and winner detection.

Board values:

-   `0`: empty cell
-   `1`: RL agent
-   `-1`: opponent

### `agent.py`

Contains the `RLAgent` class. It stores estimated values for board
states, selects moves using an epsilon-greedy strategy, and updates
state values using temporal-difference learning.

Main parameters:

-   `alpha`: learning rate
-   `gamma`: discount factor
-   `epsilon`: exploration probability

### `minimax.py`

Contains the `MinimaxOpponent` class. It searches possible future moves
recursively and chooses a move according to the Minimax strategy.

### `training.py`

Contains functions to train the RL agent against a random opponent and
evaluate it against the Minimax opponent.

### `main.py`

The entry point of the project. Run this file to start training and
evaluation.

## Reinforcement learning concepts used

-   **State:** The current arrangement of the Tic-Tac-Toe board.
-   **Action:** A legal cell in which the agent can place its mark.
-   **Reward:** Feedback received at the end of a game.
-   **Value function:** The agent's estimate of how good a state is.
-   **Epsilon-greedy:** A strategy that balances trying random moves
    (exploration) with choosing moves the agent currently estimates as
    best (exploitation).
-   **Temporal-difference learning:** Updating a state's estimated value
    using the next state's value and the reward.

## Important note

This is a small educational example using a dictionary-based value
table. It does **not** use a neural network, and the learned values are
not saved to disk. The agent's performance depends on training and may
not always draw against a perfect Minimax opponent.
