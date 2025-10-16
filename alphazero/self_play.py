"""
Self-play data generation for AlphaZero training
"""

import numpy as np
from .mcts import MCTS


class SelfPlay:
    """
    Self-play game generator
    Generates training data through self-play using MCTS
    """

    def __init__(self, network, game, args):
        """
        Initialize self-play

        Args:
            network: Neural network for policy and value prediction
            game: Game instance
            args: Configuration arguments
        """
        self.network = network
        self.game = game
        self.args = args

    def play_game(self, temperature_schedule=None):
        """
        Play one self-play game and collect training data

        Args:
            temperature_schedule: Function that takes move number and returns temperature
                                  Default: temperature=1 for first 30 moves, then 0

        Returns:
            list: Training examples [(state, policy, value), ...]
        """
        if temperature_schedule is None:
            temperature_schedule = lambda move_num: 1.0 if move_num < 30 else 0.0

        train_examples = []
        game = self.game.clone()
        game.reset()

        mcts = MCTS(self.network, game, self.args)
        move_count = 0

        while True:
            move_count += 1
            temperature = temperature_schedule(move_count)

            # Get canonical board (from current player's perspective)
            canonical_board = game.get_canonical_board()

            # Get MCTS policy
            action_probs = mcts.get_action_probs(game, temperature=temperature)

            # Store training example
            # Note: we store the canonical board and policy
            # The value will be filled in at the end of the game
            train_examples.append([canonical_board, action_probs, None])

            # Sample action from MCTS policy
            action = np.random.choice(len(action_probs), p=action_probs)

            # Execute action
            game.get_next_state(action)
            mcts.update_root(action)

            # Check if game ended
            game_result = game.get_game_ended()
            if game_result != 0:
                # Game ended, assign values to all training examples
                # The value is from the perspective of the current player of each position
                return self._assign_rewards(train_examples, game_result)

    def _assign_rewards(self, train_examples, game_result):
        """
        Assign rewards to training examples based on game result

        Args:
            train_examples: List of [state, policy, None]
            game_result: Final game result from last player's perspective

        Returns:
            list: Training examples with values assigned
        """
        # game_result is from the perspective of the player who just moved
        # We need to assign values from each position's current player perspective

        # The game alternates players, so we need to flip the result for each position
        completed_examples = []
        result = game_result

        for i in reversed(range(len(train_examples))):
            state, policy, _ = train_examples[i]
            # Assign value from current player's perspective at this position
            completed_examples.append([state, policy, result])
            # Flip result for the next (previous) position
            result = -result

        completed_examples.reverse()
        return completed_examples

    def generate_games(self, num_games):
        """
        Generate multiple self-play games

        Args:
            num_games: Number of games to generate

        Returns:
            list: All training examples from all games
        """
        all_examples = []

        for game_num in range(num_games):
            examples = self.play_game()
            all_examples.extend(examples)

            if (game_num + 1) % max(1, num_games // 10) == 0:
                print(f"  Generated {game_num + 1}/{num_games} games")

        return all_examples


def augment_data(train_examples, game):
    """
    Augment training data with symmetries

    Args:
        train_examples: List of [state, policy, value]
        game: Game instance for getting symmetries

    Returns:
        list: Augmented training examples
    """
    augmented_examples = []

    for state, policy, value in train_examples:
        # Get all symmetries
        symmetries = game.get_symmetries(state, policy)

        for sym_state, sym_policy in symmetries:
            augmented_examples.append([sym_state, sym_policy, value])

    return augmented_examples
