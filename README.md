# AlphaZero Gomoku

A reinforcement learning project implementing AlphaZero for Gomoku (Five in a Row). This project supports multiple board sizes (5×5, 10×10, 15×15) for progressive training and learning.

## 📚 Documentation

**新手？** 我们为零基础学习者准备了完整的中文技术文档！

👉 **[查看完整新手教程](./docs/)** - 从零开始学习AlphaZero

文档包含：
- 🎓 基础概念（像教小朋友一样讲解）
- 📁 项目结构详解
- 🧠 核心算法原理（MCTS、ResNet、自我对弈）
- 🚀 训练全流程（手把手教学）
- 💻 代码走读（逐行讲解）
- 🎯 实战练习（动手训练）

## Features

- **AlphaZero Algorithm**: Complete implementation with MCTS and deep neural networks
- **Multi-size Support**: Train on 5×5, 10×10, or 15×15 boards
- **Cross-platform**: Works on both Mac (CPU) and Windows (CUDA GPU)
- **Self-play Training**: Generates training data through self-play
- **Human vs AI**: Play against trained models
- **Model Evaluation**: Evaluate models against random baseline or compare models

## Project Structure

```
Gomoku_RL/
├── gomoku/              # Game logic
│   ├── board.py         # Board implementation
│   └── game.py          # Game wrapper
├── alphazero/           # AlphaZero implementation
│   ├── network.py       # ResNet policy-value network
│   ├── mcts.py          # Monte Carlo Tree Search
│   ├── self_play.py     # Self-play data generation
│   └── trainer.py       # Training loop
├── configs/             # Configuration files
│   ├── config_5x5.py    # 5×5 board config
│   ├── config_10x10.py  # 10×10 board config
│   └── config_15x15.py  # 15×15 board config
├── utils/               # Utility functions
│   ├── device.py        # CUDA/CPU detection
│   └── logger.py        # Logging utilities
├── train.py             # Main training script
├── play.py              # Human vs AI interface
├── evaluate.py          # Model evaluation
└── check_env.py         # Environment check
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training on Windows)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Gomoku_RL.git
cd Gomoku_RL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Check your environment:
```bash
python check_env.py
```

## Usage

### 1. Training

Train a model on a specific board size:

```bash
# Quick training on 5×5 (for testing)
python train.py --board_size 5

# Medium training on 10×10
python train.py --board_size 10

# Full training on 15×15 (requires GPU)
python train.py --board_size 15
```

Additional options:
```bash
# Specify number of iterations
python train.py --board_size 5 --iterations 50

# Resume from checkpoint
python train.py --board_size 10 --resume models/10x10/checkpoint_iter_50.pth

# Force CPU training
python train.py --board_size 5 --cpu
```

### 2. Play Against AI

Play against a trained model:

```bash
python play.py --checkpoint models/5x5/checkpoint_latest.pth --board_size 5

# Play as black (first)
python play.py --checkpoint models/10x10/checkpoint_latest.pth --board_size 10 --human_first

# Adjust AI strength (fewer simulations = weaker)
python play.py --checkpoint models/15x15/checkpoint_latest.pth --board_size 15 --simulations 200
```

### 3. Evaluate Models

Evaluate model against random player:

```bash
python evaluate.py --checkpoint models/5x5/checkpoint_latest.pth --board_size 5 --num_games 100
```

Compare two models:

```bash
python evaluate.py --checkpoint models/10x10/checkpoint_iter_100.pth \
                   --checkpoint2 models/10x10/checkpoint_iter_50.pth \
                   --baseline model --board_size 10 --num_games 50
```

## Training Strategy

### Phase 1: 5×5 Board (Quick Validation)
- **Purpose**: Verify algorithm correctness
- **Time**: A few hours on CPU
- **Configuration**: 4 ResNet blocks, 200 MCTS simulations
- **Expected**: Agent learns basic patterns

### Phase 2: 10×10 Board (Medium Scale)
- **Purpose**: Test scalability
- **Time**: 1-2 days on GPU
- **Configuration**: 8 ResNet blocks, 400 MCTS simulations
- **Expected**: More sophisticated strategies

### Phase 3: 15×15 Board (Full Gomoku)
- **Purpose**: Achieve strong AI performance
- **Time**: Several days on GPU
- **Configuration**: 10 ResNet blocks, 800 MCTS simulations
- **Expected**: Near-expert level play

## Configuration

Each board size has its own configuration file in `configs/`:

- `config_5x5.py`: Fast training, suitable for CPU
- `config_10x10.py`: Medium training, GPU recommended
- `config_15x15.py`: Full training, GPU required

Key parameters you can adjust:
- `NUM_SIMULATIONS`: MCTS simulations per move
- `NUM_SELF_PLAY_GAMES`: Games per training iteration
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Neural network learning rate
- `NUM_RES_BLOCKS`: Number of residual blocks in network

## Cross-Platform Training

### Development on Mac (CPU)
```bash
# Quick iterations on 5×5
python train.py --board_size 5 --iterations 20
```

### Training on Windows (GPU)
```bash
# Full training on larger boards
python train.py --board_size 15 --iterations 500
```

The code automatically detects CUDA availability and uses GPU when available.

## Algorithm Details

### AlphaZero Components

1. **Neural Network**: ResNet architecture with two heads
   - Policy head: Outputs move probabilities
   - Value head: Outputs position evaluation

2. **MCTS**: Monte Carlo Tree Search with UCB selection
   - Upper Confidence Bound formula: Q + c_puct × P × √N_parent / (1 + N)
   - Dirichlet noise at root for exploration

3. **Self-play**: Generate training data
   - Play games using current network + MCTS
   - Store (state, policy, outcome) tuples
   - Data augmentation with symmetries (rotations, flips)

4. **Training**: Update neural network
   - Policy loss: Cross-entropy between MCTS policy and network policy
   - Value loss: MSE between game outcome and network value
   - Combined loss optimized with Adam

## Tips for Training

1. **Start Small**: Always test on 5×5 first to verify everything works
2. **Monitor Loss**: Training loss should decrease steadily
3. **Save Checkpoints**: Models are saved every N iterations
4. **GPU Memory**: Adjust batch size if running out of memory
5. **Training Time**: Be patient - good models take time to train

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in config
- Reduce `NUM_SIMULATIONS` for MCTS
- Use smaller network (fewer `NUM_RES_BLOCKS`)

### Training Too Slow
- Ensure CUDA is being used (check with `python check_env.py`)
- Reduce `NUM_SELF_PLAY_GAMES`
- Use smaller board size for testing

### Model Not Learning
- Increase `NUM_SIMULATIONS` for better MCTS policies
- Check loss curves - they should decrease
- Train for more iterations
- Try different learning rates

## Future Improvements

- [ ] Parallel self-play with multiprocessing
- [ ] Learning rate scheduling
- [ ] Arena evaluation (new model vs old model)
- [ ] Web interface for playing
- [ ] Pre-trained models for download
- [ ] TensorBoard integration for monitoring

## References

- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [MCTS Survey](https://ieeexplore.ieee.org/document/6145622)

## License

MIT License - feel free to use this project for learning and experimentation.

## Acknowledgments

This project is a learning implementation of the AlphaZero algorithm for educational purposes.
