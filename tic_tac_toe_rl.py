import numpy as np
import random

# ==========================================
# 1. ENVIRONMENT: TIC-TAC-TOE GAME ENGINE
# ==========================================
class TicTacToe:
    def __init__(self):
        # Initialize a 3x3 board filled with zeros.
        # 0 = empty space, 1 = RL Agent, -1 = Opponent / Minimax
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 (RL Agent) always starts

    def reset(self):
        """Resets the board to the initial empty state for a new game episode."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """
        Converts the 3x3 NumPy board array into a flattened tuple.
        Tuples are hashable, which allows us to use them as keys in Python dictionaries.
        """
        return tuple(self.board.flatten())

    def available_actions(self):
        """Returns a list of all available (row, col) coordinates where moves can be made."""
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def step(self, action):
        """
        Executes a player's action on the board, checks for game over conditions,
        and switches the active player.
        """
        # Place the current player's marker (1 or -1) on the chosen cell
        self.board[action] = self.current_player
        
        # Check if the current move resulted in a win
        if self.check_winner(self.current_player):
            # Reward +1.0 if Agent wins, -1.0 if Opponent wins
            reward = 1.0 if self.current_player == 1 else -1.0
            return self.get_state(), reward, True
        
        # Check if the board is full (Draw)
        if len(self.available_actions()) == 0:
            return self.get_state(), 0.0, True
            
        # Switch turn to the other player (multiplies by -1 to toggle between 1 and -1)
        self.current_player *= -1
        return self.get_state(), 0.0, False

    def check_winner(self, player):
        """Checks rows, columns, and both diagonals for a winning line for the given player."""
        # Check rows and columns
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        # Check main and anti-diagonals
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False


# ==========================================
# 2. REINFORCEMENT LEARNING AGENT (TD LEARNING)
# ==========================================
class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.values = {}       # Value table mapping state tuples to expected future rewards V(s)
        self.alpha = alpha     # Learning rate: determines how much new updates override old values
        self.gamma = gamma     # Discount factor: weights future rewards vs immediate rewards
        self.epsilon = epsilon # Exploration rate: probability of making a random move to explore

    def get_value(self, state):
        """Returns the learned value of a state. Defaults to 0.0 for unseen states."""
        return self.values.get(state, 0.0)

    def choose_action(self, board, available_actions):
        """
        Selects an action using an epsilon-greedy policy:
        - With probability epsilon, explore by picking a random action.
        - Otherwise, exploit learned knowledge by picking the action leading to the highest value state.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        
        best_value = -float('inf')
        best_action = available_actions[0]
        
        # Evaluate all possible next states and pick the best one
        for action in available_actions:
            next_board = board.copy()
            next_board[action] = 1  # Assuming agent is player 1
            next_state = tuple(next_board.flatten())
            value = self.get_value(next_state)
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action

    def update_value(self, state, reward, next_state):
        """
        Updates the value estimate of a state using Temporal Difference (TD) learning:
        V(s) = V(s) + alpha * [reward + gamma * V(next_state) - V(s)]
        """
        self.values[state] = self.get_value(state) + self.alpha * (
            reward + self.gamma * self.get_value(next_state) - self.get_value(state)
        )


# ==========================================
# 3. MINIMAX OPPONENT (UNBEATABLE BASELINE)
# ==========================================
class MinimaxOpponent:
    def evaluate(self, board):
        """Scores the board from Minimax's perspective (-1): wins are negative, agent wins are positive."""
        for i in range(3):
            if np.all(board[i, :] == -1) or np.all(board[:, i] == -1):
                return -10
            if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
                return 10
        if np.all(np.diag(board) == -1) or np.all(np.diag(np.fliplr(board)) == -1):
            return -10
        if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
            return 10
        return 0

    def minimax(self, board, depth, is_maximizing):
        """Recursive Minimax algorithm evaluating all future outcomes to play perfectly."""
        score = self.evaluate(board)
        
        # Terminal states with depth penalty/bonus to favor faster wins or slower losses
        if score == 10: return score - depth
        if score == -10: return score + depth
        if not 0 in board: return 0  # Draw

        if is_maximizing:
            max_eval = -float('inf')
            for r in range(3):
                for c in range(3):
                    if board[r, c] == 0:
                        board[r, c] = 1
                        eval = self.minimax(board, depth + 1, False)
                        board[r, c] = 0
                        max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for r in range(3):
                for c in range(3):
                    if board[r, c] == 0:
                        board[r, c] = -1
                        eval = self.minimax(board, depth + 1, True)
                        board[r, c] = 0
                        min_eval = min(min_eval, eval)
            return min_eval

    def get_action(self, board):
        """Finds the mathematically optimal move for the opponent (-1)."""
        best_val = float('inf')
        best_action = None
        for r in range(3):
            for c in range(3):
                if board[r, c] == 0:
                    board[r, c] = -1
                    move_val = self.minimax(board, 0, True)
                    board[r, c] = 0
                    if move_val < best_val:
                        best_val = move_val
                        best_action = (r, c)
        return best_action


# ==========================================
# 4. TRAINING LOOP (SELF-PLAY)
# ==========================================
def train_agent(episodes=40000):
    env = TicTacToe()
    agent = RLAgent()

    print(f"Training agent via self-play for {episodes} episodes...")
    for episode in range(episodes):
        state = env.reset()
        done = false_flag = False
        state_history = []  # Tracks states encountered during the episode for backward updating

        while not done:
            actions = env.available_actions()
            if not actions:
                break
                
            # Alternate turns: Agent (Player 1) uses RL policy, Opponent uses random moves
            if env.current_player == 1:
                action = agent.choose_action(env.board, actions)
            else:
                action = random.choice(actions)
                
            next_state, reward, done = env.step(action)
            
            # Record state transition history specifically for the RL agent's turns
            if env.current_player == 1:
                state_history.append((state, reward, next_state))
                
            state = next_state

        # Backward propagation: update value estimations from the end of the game back to the start
        for state, reward, next_state in reversed(state_history):
            agent.update_value(state, reward, next_state)

        # Periodically decay epsilon to reduce random exploration over time
        if episode > 0 and episode % 5000 == 0:
            agent.epsilon = max(0.05, agent.epsilon * 0.9)

    print("Training complete!\n")
    return agent


# ==========================================
# 5. EVALUATION LOOP (AGAINST MINIMAX)
# ==========================================
def evaluate_agent(agent, num_games=1000):
    env = TicTacToe()
    minimax = MinimaxOpponent()
    
    wins, losses, draws = 0, 0, 0
    agent.epsilon = 0.0  # Turn off exploration completely for deterministic evaluation

    print(f"Evaluating agent against Minimax opponent for {num_games} games...")
    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            actions = env.available_actions()
            if not actions:
                break

            # Agent plays as Player 1, Minimax plays as Player -1
            if env.current_player == 1:
                action = agent.choose_action(env.board, actions)
            else:
                action = minimax.get_action(env.board)

            _, reward, done = env.step(action)

            # Tally game outcomes
            if done:
                if reward == 1.0:
                    wins += 1
                elif reward == -1.0:
                    losses += 1
                else:
                    draws += 1

    print(f"\n--- Evaluation Results ---")
    print(f"Wins:   {wins} ({wins/num_games*100}%)")
    print(f"Losses: {losses} ({losses/num_games*100}%)")
    print(f"Draws:  {draws} ({draws/num_games*100}%)")


if __name__ == "__main__":
    trained_agent = train_agent(episodes=40000)
    evaluate_agent(trained_agent, num_games=1000)