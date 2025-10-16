"""
Game wrapper for AlphaZero
Provides unified interface for self-play and training
"""

import numpy as np
from .board import GomokuBoard


class GomokuGame:
    """
    Gomoku game wrapper for AlphaZero
    Handles game logic and provides standardized interface
    """

    def __init__(self, board_size=15, n_in_row=5):
        """
        Initialize game

        Args:
            board_size: Size of the board (5, 10, or 15)
            n_in_row: Number in a row needed to win
        """
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.board = GomokuBoard(board_size, n_in_row)

    def reset(self):
        """
        Reset game to initial state

        Returns:
            numpy.ndarray: Initial board state
        """
        self.board.reset()
        return self.board.get_feature_planes()

    def get_board_size(self):
        """Get board size"""
        return self.board_size

    def get_action_size(self):
        """
        Get total number of possible actions

        Returns:
            int: board_size * board_size
        """
        return self.board_size * self.board_size

    def get_next_state(self, action):
        """
        Execute action and get next state

        Args:
            action: Action index (0 to board_size^2 - 1)

        Returns:
            tuple: (next_state, current_player)
        """
        row, col = self._action_to_position(action)

        if not self.board.make_move(row, col):
            raise ValueError(f"Invalid action: {action} at position ({row}, {col})")

        next_state = self.board.get_feature_planes()
        current_player = self.board.current_player

        return next_state, current_player

    def get_valid_moves(self):
        """
        Get valid moves as a binary vector

        Returns:
            numpy.ndarray: Binary vector of size action_size
        """
        valid_moves = np.zeros(self.get_action_size(), dtype=np.float32)

        for row, col in self.board.get_valid_moves():
            action = self._position_to_action(row, col)
            valid_moves[action] = 1

        return valid_moves

    def get_game_ended(self):
        """
        Check if game has ended and get winner

        Returns:
            float: 0 if game ongoing,
                   1 if current player won,
                   -1 if current player lost,
                   0.0001 for draw (small value to distinguish from ongoing)
        """
        game_over, winner = self.board.get_game_status()

        if not game_over:
            return 0

        if winner == 0:  # Draw
            return 0.0001  # Small non-zero value to distinguish from ongoing game

        # Return result from current player's perspective
        # If winner matches current player, previous player (opponent) won
        # So current player lost
        if winner == self.board.current_player:
            return -1  # Current player lost (because opponent just won)
        else:
            return 1  # Current player won

    def get_canonical_board(self):
        """
        Get board from current player's perspective

        Returns:
            numpy.ndarray: Feature planes from current player's view
        """
        return self.board.get_feature_planes()

    def get_symmetries(self, board_state, policy):
        """
        Get all symmetries of the board and policy
        Used for data augmentation

        Args:
            board_state: Board state array (3, board_size, board_size)
            policy: Policy vector (board_size * board_size)

        Returns:
            list: List of (board, policy) tuples for all symmetries
        """
        symmetries = []

        # Convert policy vector to 2D
        policy_2d = policy.reshape(self.board_size, self.board_size)

        # Original
        symmetries.append((board_state, policy))

        # Rotation 90, 180, 270
        for k in range(1, 4):
            rotated_board = np.rot90(board_state, k, axes=(1, 2))
            rotated_policy = np.rot90(policy_2d, k)
            symmetries.append((rotated_board, rotated_policy.flatten()))

        # Horizontal flip
        flipped_board = np.flip(board_state, axis=2)
        flipped_policy = np.flip(policy_2d, axis=1)
        symmetries.append((flipped_board.copy(), flipped_policy.flatten().copy()))

        # Horizontal flip + rotations
        for k in range(1, 4):
            rotated_board = np.rot90(flipped_board, k, axes=(1, 2))
            rotated_policy = np.rot90(flipped_policy, k)
            symmetries.append((rotated_board.copy(), rotated_policy.flatten().copy()))

        return symmetries

    def display(self):
        """Display the current board"""
        self.board.display()

    def _position_to_action(self, row, col):
        """
        Convert (row, col) position to action index

        Args:
            row: Row index
            col: Column index

        Returns:
            int: Action index
        """
        return row * self.board_size + col

    def _action_to_position(self, action):
        """
        Convert action index to (row, col) position

        Args:
            action: Action index

        Returns:
            tuple: (row, col)
        """
        row = action // self.board_size
        col = action % self.board_size
        return row, col

    def clone(self):
        """
        Create a deep copy of the game

        Returns:
            GomokuGame: A copy of the current game
        """
        new_game = GomokuGame(self.board_size, self.n_in_row)
        new_game.board = self.board.copy()
        return new_game

    def __str__(self):
        return f"GomokuGame(size={self.board_size}, n_in_row={self.n_in_row})"

    def __repr__(self):
        return self.__str__()
