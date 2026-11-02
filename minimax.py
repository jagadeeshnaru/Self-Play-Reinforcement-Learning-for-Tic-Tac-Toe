import numpy as np


class MinimaxOpponent:
    def evaluate(self, board):
        # Negative scores favor Minimax; positive scores favor the RL agent.
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
        score = self.evaluate(board)

        # Stop searching when someone has won or the board is full.
        if score == 10:
            return score - depth
        if score == -10:
            return score + depth
        if not np.any(board == 0):
            return 0

        if is_maximizing:
            best_score = -float("inf")

            for row in range(3):
                for col in range(3):
                    if board[row, col] == 0:
                        board[row, col] = 1
                        score = self.minimax(board, depth + 1, False)
                        board[row, col] = 0
                        best_score = max(best_score, score)

            return best_score

        best_score = float("inf")

        for row in range(3):
            for col in range(3):
                if board[row, col] == 0:
                    board[row, col] = -1
                    score = self.minimax(board, depth + 1, True)
                    board[row, col] = 0
                    best_score = min(best_score, score)

        return best_score

    def get_action(self, board):
        # Minimax plays as -1, so it chooses the move with the lowest score.
        best_score = float("inf")
        best_action = None

        for row in range(3):
            for col in range(3):
                if board[row, col] == 0:
                    board[row, col] = -1
                    score = self.minimax(board, 0, True)
                    board[row, col] = 0

                    if score < best_score:
                        best_score = score
                        best_action = (row, col)

        return best_action
