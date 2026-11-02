import random


class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        # Map each board state to its estimated future reward.
        self.values = {}

        # Learning rate, discount factor, and exploration probability.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_value(self, state):
        # Unseen states start with value 0.
        return self.values.get(state, 0.0)

    def choose_action(self, board, available_actions):
        # Explore: sometimes choose a random legal move.
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)

        # Exploit: choose the move with the highest learned state value.
        best_value = -float("inf")
        best_action = available_actions[0]

        for action in available_actions:
            next_board = board.copy()
            next_board[action] = 1
            next_state = tuple(next_board.flatten())
            value = self.get_value(next_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def update_value(self, state, reward, next_state=None, terminal=False):
        # TD target is the final reward for a terminal transition.
        current_value = self.get_value(state)

        if terminal:
            target = reward
        else:
            target = reward + self.gamma * self.get_value(next_state)

        # Move the old estimate a little closer to the new target.
        td_error = target - current_value
        self.values[state] = current_value + self.alpha * td_error
