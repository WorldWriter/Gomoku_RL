"""
Configuration for 5x5 Gomoku
Fast training for algorithm verification
"""

# Game settings
BOARD_SIZE = 5
N_IN_ROW = 4  # 4-in-a-row for 5x5 board

# Network architecture
NUM_CHANNELS = 64
NUM_RES_BLOCKS = 4

# MCTS settings
NUM_SIMULATIONS = 200  # Reduced for faster training
C_PUCT = 1.0  # Exploration constant

# Self-play settings
NUM_SELF_PLAY_GAMES = 50  # Games per iteration
TEMPERATURE_THRESHOLD = 15  # Use temperature=1 for first N moves

# Training settings
NUM_ITERATIONS = 100
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Data management
MAX_HISTORY_LEN = 50000  # Maximum training examples to keep
USE_AUGMENTATION = True  # Use symmetry augmentation

# Checkpoint settings
CHECKPOINT_DIR = 'models/5x5'
SAVE_INTERVAL = 10  # Save every N iterations


def get_config():
    """Get configuration dictionary"""
    return {
        # Game
        'board_size': BOARD_SIZE,
        'n_in_row': N_IN_ROW,

        # Network
        'num_channels': NUM_CHANNELS,
        'num_res_blocks': NUM_RES_BLOCKS,

        # MCTS
        'num_simulations': NUM_SIMULATIONS,
        'c_puct': C_PUCT,

        # Self-play
        'num_self_play_games': NUM_SELF_PLAY_GAMES,
        'temperature_threshold': TEMPERATURE_THRESHOLD,

        # Training
        'num_iterations': NUM_ITERATIONS,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,

        # Data
        'max_history_len': MAX_HISTORY_LEN,
        'use_augmentation': USE_AUGMENTATION,

        # Checkpoint
        'checkpoint_dir': CHECKPOINT_DIR,
        'save_interval': SAVE_INTERVAL,
    }
