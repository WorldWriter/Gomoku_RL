"""
Model evaluation script
Evaluate AlphaZero models through self-play or against baselines
"""

import sys
# 添加E盘PyTorch安装路径
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gomoku import GomokuGame
from alphazero import AlphaZeroNet, MCTS
from utils import get_device


class RandomPlayer:
    """Random baseline player"""

    def __init__(self, game):
        self.game = game

    def get_action(self, game):
        """
        Get random valid action

        Args:
            game: Current game state

        Returns:
            int: Action index
        """
        valid_moves = game.get_valid_moves()
        valid_actions = np.where(valid_moves > 0)[0]
        return np.random.choice(valid_actions)


class AlphaZeroPlayer:
    """AlphaZero player using MCTS"""

    def __init__(self, network, game, num_simulations=400):
        self.network = network
        self.game = game
        self.mcts = MCTS(
            network,
            game,
            {'num_simulations': num_simulations, 'c_puct': 1.0}
        )

    def get_action(self, game):
        """
        Get action using MCTS

        Args:
            game: Current game state

        Returns:
            int: Action index
        """
        action_probs = self.mcts.get_action_probs(game, temperature=0)
        return np.argmax(action_probs)


def play_game(player1, player2, game, verbose=False):
    """
    Play one game between two players

    Args:
        player1: First player (plays as black)
        player2: Second player (plays as white)
        game: Game instance
        verbose: Print game progress

    Returns:
        int: Game result (1: player1 wins, -1: player2 wins, 0: draw)
    """
    game.reset()
    current_player = player1
    other_player = player2

    move_count = 0

    while True:
        if verbose:
            print(f"\nMove {move_count + 1}")
            game.display()

        # Get action
        action = current_player.get_action(game)

        # Execute action
        game.get_next_state(action)
        move_count += 1

        # Update MCTS tree if applicable
        if isinstance(other_player, AlphaZeroPlayer):
            other_player.mcts.update_root(action)

        # Check if game ended
        game_result = game.get_game_ended()
        if game_result != 0:
            if verbose:
                print("\nGame Over!")
                game.display()
                if abs(game_result) < 0.001:
                    print("Result: Draw")
                else:
                    winner = "Player 1" if game_result > 0 else "Player 2"
                    print(f"Winner: {winner}")

            return game_result

        # Switch players
        current_player, other_player = other_player, current_player


def evaluate_against_random(model_path, board_size, num_games=100, simulations=400):
    """
    Evaluate model against random player

    Args:
        model_path: Path to model checkpoint
        board_size: Board size
        num_games: Number of games to play
        simulations: MCTS simulations per move

    Returns:
        dict: Evaluation results
    """
    device = get_device()

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if 'args' in checkpoint:
        config = checkpoint['args']
        network = AlphaZeroNet(
            board_size=board_size,
            num_channels=config.get('num_channels', 128),
            num_res_blocks=config.get('num_res_blocks', 10)
        )
    else:
        network = AlphaZeroNet(board_size=board_size)

    network.load_state_dict(checkpoint['model_state_dict'])
    network = network.to(device)
    network.eval()

    print(f"Model loaded. Iteration: {checkpoint.get('iteration', 'unknown')}")

    # Create game and players
    game = GomokuGame(board_size=board_size)
    ai_player = AlphaZeroPlayer(network, game, num_simulations=simulations)
    random_player = RandomPlayer(game)

    # Play games
    results = {'ai_wins': 0, 'random_wins': 0, 'draws': 0}

    print(f"\nPlaying {num_games} games against random player...")
    print(f"MCTS simulations: {simulations}")

    for game_num in tqdm(range(num_games)):
        # Alternate who plays first
        if game_num % 2 == 0:
            result = play_game(ai_player, random_player, game)
            if result > 0:
                results['ai_wins'] += 1
            elif result < 0:
                results['random_wins'] += 1
            else:
                results['draws'] += 1
        else:
            result = play_game(random_player, ai_player, game)
            if result > 0:
                results['random_wins'] += 1
            elif result < 0:
                results['ai_wins'] += 1
            else:
                results['draws'] += 1

    # Calculate statistics
    win_rate = results['ai_wins'] / num_games * 100
    draw_rate = results['draws'] / num_games * 100
    loss_rate = results['random_wins'] / num_games * 100

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total games: {num_games}")
    print(f"AI wins: {results['ai_wins']} ({win_rate:.1f}%)")
    print(f"Random wins: {results['random_wins']} ({loss_rate:.1f}%)")
    print(f"Draws: {results['draws']} ({draw_rate:.1f}%)")
    print("="*60)

    return results


def compare_models(model1_path, model2_path, board_size, num_games=100, simulations=400):
    """
    Compare two models by playing against each other

    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        board_size: Board size
        num_games: Number of games to play
        simulations: MCTS simulations per move

    Returns:
        dict: Comparison results
    """
    device = get_device()

    # Load models
    print(f"Loading model 1 from {model1_path}...")
    checkpoint1 = torch.load(model1_path, map_location=device)
    network1 = AlphaZeroNet(board_size=board_size)
    network1.load_state_dict(checkpoint1['model_state_dict'])
    network1 = network1.to(device)
    network1.eval()

    print(f"Loading model 2 from {model2_path}...")
    checkpoint2 = torch.load(model2_path, map_location=device)
    network2 = AlphaZeroNet(board_size=board_size)
    network2.load_state_dict(checkpoint2['model_state_dict'])
    network2 = network2.to(device)
    network2.eval()

    # Create players
    game = GomokuGame(board_size=board_size)
    player1 = AlphaZeroPlayer(network1, game, num_simulations=simulations)
    player2 = AlphaZeroPlayer(network2, game, num_simulations=simulations)

    # Play games
    results = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}

    print(f"\nPlaying {num_games} games between models...")

    for game_num in tqdm(range(num_games)):
        # Alternate who plays first
        if game_num % 2 == 0:
            result = play_game(player1, player2, game)
            if result > 0:
                results['model1_wins'] += 1
            elif result < 0:
                results['model2_wins'] += 1
            else:
                results['draws'] += 1
        else:
            result = play_game(player2, player1, game)
            if result > 0:
                results['model2_wins'] += 1
            elif result < 0:
                results['model1_wins'] += 1
            else:
                results['draws'] += 1

    # Print results
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"\nTotal games: {num_games}")
    print(f"Model 1 wins: {results['model1_wins']} ({results['model1_wins']/num_games*100:.1f}%)")
    print(f"Model 2 wins: {results['model2_wins']} ({results['model2_wins']/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--baseline', type=str, default='random',
                        choices=['random', 'model'],
                        help='Baseline to evaluate against')
    parser.add_argument('--checkpoint2', type=str, default=None,
                        help='Path to second model (for model comparison)')
    parser.add_argument('--board_size', type=int, default=15, choices=[5, 10, 15],
                        help='Board size')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of evaluation games')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move')

    args = parser.parse_args()

    if args.baseline == 'random':
        evaluate_against_random(
            args.checkpoint,
            args.board_size,
            args.num_games,
            args.simulations
        )
    elif args.baseline == 'model':
        if args.checkpoint2 is None:
            print("Error: --checkpoint2 required for model comparison")
            return
        compare_models(
            args.checkpoint,
            args.checkpoint2,
            args.board_size,
            args.num_games,
            args.simulations
        )


if __name__ == '__main__':
    main()
