"""
Gomoku Board implementation
Supports configurable board sizes: 5x5, 10x10, 15x15
"""

import numpy as np


class GomokuBoard:
    """
    Gomoku board with configurable size
    Supports n-in-a-row win condition (default: 5)
    """

    def __init__(self, size=15, n_in_row=5):
        """
        Initialize board

        Args:
            size: Board size (5, 10, or 15)
            n_in_row: Number of consecutive stones needed to win (default: 5)
        """
        self.size = size
        self.n_in_row = min(n_in_row, size)  # Ensure n_in_row doesn't exceed board size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0: empty, 1: black, -1: white
        self.current_player = 1  # 1: black (first), -1: white
        self.last_move = None
        self.move_history = []

    def reset(self):
        """Reset board to initial state"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.move_history = []

    def get_valid_moves(self):
        """
        Get all valid moves (empty positions)

        Returns:
            List of (row, col) tuples
        """
        return list(zip(*np.where(self.board == 0)))

    def get_valid_moves_mask(self):
        """
        Get valid moves as a binary mask

        Returns:
            numpy.ndarray: Binary mask (1 for valid, 0 for invalid)
        """
        return (self.board == 0).astype(np.float32)

    def is_valid_move(self, row, col):
        """
        Check if a move is valid

        Args:
            row: Row index
            col: Column index

        Returns:
            bool: True if move is valid
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        return self.board[row, col] == 0

    def make_move(self, row, col):
        """
        Make a move at the specified position

        Args:
            row: Row index
            col: Column index

        Returns:
            bool: True if move was successful
        """
        if not self.is_valid_move(row, col):
            return False

        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_history.append((row, col, self.current_player))
        self.current_player = -self.current_player

        return True

    def undo_move(self):
        """
        Undo the last move

        Returns:
            bool: True if undo was successful
        """
        if not self.move_history:
            return False

        row, col, player = self.move_history.pop()
        self.board[row, col] = 0
        self.current_player = player

        if self.move_history:
            self.last_move = (self.move_history[-1][0], self.move_history[-1][1])
        else:
            self.last_move = None

        return True

    def check_win(self, row, col):
        """
        Check if the last move at (row, col) resulted in a win

        Args:
            row: Row index of last move
            col: Column index of last move

        Returns:
            bool: True if the move resulted in a win
        """
        player = self.board[row, col]
        if player == 0:
            return False

        # Check four directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # diagonal
            [(1, -1), (-1, 1)]   # anti-diagonal
        ]

        for direction_pair in directions:
            count = 1  # Count the stone at (row, col)

            # Check both directions
            for dr, dc in direction_pair:
                r, c = row + dr, col + dc
                while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc

            if count >= self.n_in_row:
                return True

        return False

    def is_full(self):
        """
        Check if the board is full

        Returns:
            bool: True if board is full
        """
        return np.all(self.board != 0)

    def get_game_status(self):
        """
        Get current game status

        Returns:
            tuple: (game_over: bool, winner: int)
                   winner: 1 (black wins), -1 (white wins), 0 (draw/ongoing)
        """
        if self.last_move is None:
            return False, 0

        # Check if last move resulted in a win
        row, col = self.last_move
        if self.check_win(row, col):
            winner = self.board[row, col]
            return True, winner

        # Check for draw (board full)
        if self.is_full():
            return True, 0

        return False, 0

    def get_board_state(self):
        """
        Get the current board state

        Returns:
            numpy.ndarray: Copy of the board state
        """
        return self.board.copy()

    def get_canonical_board(self):
        """
        Get board from current player's perspective
        Current player's stones are always represented as 1

        Returns:
            numpy.ndarray: Canonical board state
        """
        return self.board * self.current_player

    def get_feature_planes(self):
        """
        Get feature planes for neural network input
        Returns 3 channels: current player stones, opponent stones, turn indicator

        Returns:
            numpy.ndarray: Shape (3, size, size)
        """
        current_player_plane = (self.board == self.current_player).astype(np.float32)
        opponent_plane = (self.board == -self.current_player).astype(np.float32)
        turn_plane = np.full((self.size, self.size),
                            (self.current_player + 1) / 2, dtype=np.float32)

        return np.stack([current_player_plane, opponent_plane, turn_plane], axis=0)

    def display(self):
        """Display the board in a readable format"""
        symbols = {0: '.', 1: 'X', -1: 'O'}

        # Print column numbers
        print('   ' + ' '.join(f'{i:2d}' for i in range(self.size)))

        for i in range(self.size):
            # Print row number and board content
            row_str = f'{i:2d} '
            for j in range(self.size):
                symbol = symbols[self.board[i, j]]
                # Highlight last move
                if self.last_move == (i, j):
                    row_str += f'[{symbol}]'
                else:
                    row_str += f' {symbol} '
            print(row_str)

    def copy(self):
        """
        Create a deep copy of the board

        Returns:
            GomokuBoard: A copy of the current board
        """
        new_board = GomokuBoard(self.size, self.n_in_row)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        new_board.move_history = self.move_history.copy()
        return new_board

    def __str__(self):
        """String representation of the board"""
        return f"GomokuBoard(size={self.size}, n_in_row={self.n_in_row}, moves={len(self.move_history)})"

    def __repr__(self):
        return self.__str__()
