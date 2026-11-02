import numpy as np


class TicTacToe:
    def __init__(self):
        # 0 = empty, 1 = RL agent, -1 = opponent
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        # Start a new game with an empty board.
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        # A tuple can be used as a dictionary key in the agent's value table.
        return tuple(self.board.flatten())

    def available_actions(self):
        # Return the coordinates of every empty cell.
        return [
            (row, col)
            for row in range(3)
            for col in range(3)
            if self.board[row, col] == 0
        ]

    def step(self, action):
        # Place the current player's mark.
        self.board[action] = self.current_player

        # Check whether this move wins the game.
        if self.check_winner(self.current_player):
            reward = 1.0 if self.current_player == 1 else -1.0
            return self.get_state(), reward, True

        # If no empty cells remain, the game is a draw.
        if not self.available_actions():
            return self.get_state(), 0.0, True

        # Switch between player 1 and player -1.
        self.current_player *= -1
        return self.get_state(), 0.0, False

    def check_winner(self, player):
        # Check every row and column.
        for i in range(3):
            if np.all(self.board[i, :] == player):
                return True
            if np.all(self.board[:, i] == player):
                return True

        # Check the two diagonals.
        if np.all(np.diag(self.board) == player):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player):
            return True

        return False
