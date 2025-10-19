"""
Main training script for AlphaZero Gomoku
Supports multiple board sizes: 5x5, 10x10, 15x15
"""

import sys
# 添加E盘PyTorch安装路径
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')
import importlib
import argparse
from pathlib import Path

from gomoku import GomokuGame
from alphazero import AlphaZeroNet, Trainer
from utils import get_device, setup_logger


def load_config(board_size):
    """
    Load configuration for specified board size

    Args:
        board_size: Board size (5, 10, or 15)

    Returns:
        dict: Configuration dictionary
    """
    config_module = importlib.import_module(f'configs.config_{board_size}x{board_size}')
    return config_module.get_config()


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero for Gomoku')
    parser.add_argument('--board_size', type=int, default=5, choices=[5, 10, 15],
                        help='Board size (5, 10, or 15)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of training iterations (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training (ignore CUDA)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for log files')

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(
        name=f'AlphaZero_{args.board_size}x{args.board_size}',
        log_dir=args.log_dir
    )

    # Load configuration
    logger.info(f"Loading configuration for {args.board_size}x{args.board_size} board...")
    config = load_config(args.board_size)

    # Override iterations if specified
    if args.iterations is not None:
        config['num_iterations'] = args.iterations
        logger.info(f"Overriding iterations to {args.iterations}")

    # Get device
    device = get_device(prefer_cuda=not args.cpu)
    logger.info(f"Using device: {device}")

    # Create game
    logger.info("Initializing game...")
    game = GomokuGame(
        board_size=config['board_size'],
        n_in_row=config['n_in_row']
    )

    # Create network
    logger.info("Creating neural network...")
    network = AlphaZeroNet(
        board_size=config['board_size'],
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks']
    )

    # Print network info
    num_params = sum(p.numel() for p in network.parameters())
    logger.info(f"Network architecture:")
    logger.info(f"  Board size: {config['board_size']}x{config['board_size']}")
    logger.info(f"  Channels: {config['num_channels']}")
    logger.info(f"  Residual blocks: {config['num_res_blocks']}")
    logger.info(f"  Total parameters: {num_params:,}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(network, game, config, device)

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        start_iteration = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from iteration {start_iteration}")

    # Print training configuration
    logger.info("\nTraining configuration:")
    logger.info(f"  Total iterations: {config['num_iterations']}")
    logger.info(f"  Self-play games per iteration: {config['num_self_play_games']}")
    logger.info(f"  MCTS simulations: {config['num_simulations']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Checkpoint directory: {config['checkpoint_dir']}")

    # Start training
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)

    try:
        remaining_iterations = config['num_iterations'] - start_iteration
        trainer.train(remaining_iterations)

        logger.info("\n" + "="*60)
        logger.info("Training completed successfully!")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer._save_checkpoint(f"interrupted_{start_iteration}")
        logger.info("Checkpoint saved. You can resume training with --resume flag")

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
