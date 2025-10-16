"""
Human vs AI gameplay interface
Play Gomoku against trained AlphaZero agent
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from gomoku import GomokuGame
from alphazero import AlphaZeroNet, MCTS
from utils import get_device


class HumanPlayer:
    """Human player interface"""

    def __init__(self, game):
        self.game = game

    def get_action(self, game):
        """
        Get action from human input

        Args:
            game: Current game state

        Returns:
            int: Action index
        """
        board_size = game.get_board_size()

        while True:
            try:
                move_input = input("\nYour move (row col, e.g., '7 7'): ").strip()
                row, col = map(int, move_input.split())

                if not (0 <= row < board_size and 0 <= col < board_size):
                    print(f"Invalid move: position out of bounds (0-{board_size-1})")
                    continue

                action = row * board_size + col

                if game.get_valid_moves()[action] == 0:
                    print("Invalid move: position already occupied")
                    continue

                return action

            except ValueError:
                print("Invalid input format. Please enter: row col (e.g., '7 7')")
            except KeyboardInterrupt:
                print("\nGame interrupted by user")
                exit(0)


class AIPlayer:
    """AI player using AlphaZero"""

    def __init__(self, network, game, args, name="AI"):
        """
        Args:
            network: Neural network
            game: Game instance
            args: MCTS configuration
            name: Player name
        """
        self.network = network
        self.game = game
        self.args = args
        self.name = name
        self.mcts = MCTS(network, game, args)

    def get_action(self, game):
        """
        Get action from AI

        Args:
            game: Current game state

        Returns:
            int: Action index
        """
        print(f"\n{self.name} is thinking...")

        action_probs = self.mcts.get_action_probs(game, temperature=0)
        action = np.argmax(action_probs)

        # Display top moves
        top_actions = np.argsort(action_probs)[-3:][::-1]
        print(f"{self.name}'s top moves:")
        for i, a in enumerate(top_actions, 1):
            row = a // game.get_board_size()
            col = a % game.get_board_size()
            prob = action_probs[a]
            print(f"  {i}. ({row}, {col}): {prob:.3f}")

        return action


def play_game(human_player, ai_player, game, human_first=True):
    """
    Play one game between human and AI

    Args:
        human_player: Human player instance
        ai_player: AI player instance
        game: Game instance
        human_first: Whether human plays first (as black)

    Returns:
        int: Game result (1: black wins, -1: white wins, 0: draw)
    """
    game.reset()

    if human_first:
        current_player = human_player
        other_player = ai_player
        print("\nYou are playing as BLACK (X)")
        print("AI is playing as WHITE (O)")
    else:
        current_player = ai_player
        other_player = human_player
        print("\nAI is playing as BLACK (X)")
        print("You are playing as WHITE (O)")

    print(f"\nBoard size: {game.get_board_size()}x{game.get_board_size()}")
    print("Enter moves as 'row col' (e.g., '7 7')")
    print("\n" + "="*60)

    move_count = 0

    while True:
        # Display board
        print(f"\nMove {move_count + 1}")
        game.display()

        # Get action
        action = current_player.get_action(game)

        # Execute action
        row = action // game.get_board_size()
        col = action % game.get_board_size()

        player_name = "You" if current_player == human_player else current_player.name
        print(f"{player_name} played: ({row}, {col})")

        game.get_next_state(action)
        move_count += 1

        # Update MCTS tree for AI
        if isinstance(other_player, AIPlayer):
            other_player.mcts.update_root(action)

        # Check if game ended
        game_result = game.get_game_ended()
        if game_result != 0:
            print("\n" + "="*60)
            game.display()
            print("\nGame Over!")

            if abs(game_result) < 0.001:  # Draw
                print("Result: Draw")
                return 0
            else:
                # Determine winner
                if human_first:
                    # If human is black: result > 0 means black won (human won)
                    winner = "You" if game_result > 0 else ai_player.name
                else:
                    # If human is white: result > 0 means black won (AI won)
                    winner = ai_player.name if game_result > 0 else "You"

                print(f"Winner: {winner}!")
                return game_result

        # Switch players
        current_player, other_player = other_player, current_player


def main():
    parser = argparse.ArgumentParser(description='Play Gomoku against AlphaZero')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--board_size', type=int, default=15, choices=[5, 10, 15],
                        help='Board size (5, 10, or 15)')
    parser.add_argument('--n_in_row', type=int, default=5,
                        help='Number in a row to win')
    parser.add_argument('--simulations', type=int, default=400,
                        help='Number of MCTS simulations')
    parser.add_argument('--human_first', action='store_true',
                        help='Human plays first (as black)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (ignore CUDA)')

    args = parser.parse_args()

    # Get device
    device = get_device(prefer_cuda=not args.cpu)

    # Create game
    print(f"\nInitializing {args.board_size}x{args.board_size} Gomoku...")
    game = GomokuGame(board_size=args.board_size, n_in_row=args.n_in_row)

    # Load network
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Infer network architecture from checkpoint if available
    if 'args' in checkpoint:
        config = checkpoint['args']
        network = AlphaZeroNet(
            board_size=args.board_size,
            num_channels=config.get('num_channels', 128),
            num_res_blocks=config.get('num_res_blocks', 10)
        )
    else:
        # Use default architecture
        network = AlphaZeroNet(board_size=args.board_size)

    network.load_state_dict(checkpoint['model_state_dict'])
    network = network.to(device)
    network.eval()

    print(f"Model loaded successfully!")
    print(f"Training iteration: {checkpoint.get('iteration', 'unknown')}")

    # Create players
    human_player = HumanPlayer(game)

    mcts_args = {
        'num_simulations': args.simulations,
        'c_puct': 1.0,
    }
    ai_player = AIPlayer(network, game, mcts_args, name="AlphaZero")

    # Play game
    print("\n" + "="*60)
    print("Starting game...")
    print("="*60)

    try:
        play_game(human_player, ai_player, game, human_first=args.human_first)

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")

    print("\nThank you for playing!")


if __name__ == '__main__':
    main()
