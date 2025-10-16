# Quick Start Guide

## 1. First Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Check your environment
python check_env.py
```

## 2. Quick Training Test (5 minutes)

Start with a small 5×5 board to verify everything works:

```bash
# Train for just 5 iterations to test
python train.py --board_size 5 --iterations 5
```

This will:
- Create a 5×5 Gomoku game
- Initialize a small neural network
- Run 5 training iterations
- Save checkpoints to `models/5x5/`

## 3. Play Against Your Model

After training, play against the AI:

```bash
python play.py --checkpoint models/5x5/checkpoint_latest.pth --board_size 5 --human_first
```

Enter moves as "row col", e.g., "2 2" for center position.

## 4. Full Training

Once verified, start full training:

### On Mac (CPU) - 5×5 Board
```bash
python train.py --board_size 5 --iterations 100
```

### On Windows GPU - 10×10 or 15×15 Board
```bash
# Medium board
python train.py --board_size 10 --iterations 200

# Full board (requires powerful GPU)
python train.py --board_size 15 --iterations 500
```

## 5. Monitor Training

Logs are saved to `logs/` directory. Check them to monitor progress:
- Training should show decreasing loss
- Self-play games should complete successfully
- Checkpoints saved every 10 iterations

## 6. Evaluate Your Model

Test against random player:
```bash
python evaluate.py --checkpoint models/5x5/checkpoint_latest.pth --board_size 5 --num_games 50
```

Expected results after good training:
- 5×5 board: >90% win rate against random
- 10×10 board: >95% win rate against random
- 15×15 board: >98% win rate against random

## Tips

1. **Start Small**: Always test on 5×5 first
2. **GPU Training**: Use Windows GPU for larger boards
3. **Training Time**:
   - 5×5: Few hours on CPU
   - 10×10: 1-2 days on GPU
   - 15×15: 3-7 days on GPU
4. **Checkpoint Often**: Models are auto-saved, you can resume with `--resume`

## Troubleshooting

### Import Error
```bash
# Make sure you're in the project directory
cd Gomoku_RL
python train.py --board_size 5
```

### Out of Memory
```bash
# Use CPU mode
python train.py --board_size 5 --cpu

# Or edit configs to reduce batch_size
```

### Training Too Slow
- Reduce NUM_SELF_PLAY_GAMES in config files
- Reduce NUM_SIMULATIONS in config files
- Use smaller board size

## Next Steps

After successful training:
1. Compare models from different iterations
2. Experiment with hyperparameters
3. Try transfer learning (load 5×5 model for 10×10 training)
4. Implement additional features from the README

Enjoy training your AlphaZero agent!
