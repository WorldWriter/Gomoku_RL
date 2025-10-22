"""检查中心位置"""
import sys
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')
import torch
import numpy as np
from gomoku import GomokuGame
from alphazero import AlphaZeroNet
from utils import get_device

device = get_device()
checkpoint = torch.load('models/5x5/checkpoint_latest.pth', map_location=device)
network = AlphaZeroNet(
    board_size=5,
    num_channels=checkpoint['args'].get('num_channels', 64),
    num_res_blocks=checkpoint['args'].get('num_res_blocks', 4)
)
network.load_state_dict(checkpoint['model_state_dict'])
network = network.to(device)
network.eval()

game = GomokuGame(board_size=5)
game.reset()

policy, value = network.predict(game.get_canonical_board())
valid_moves = game.get_valid_moves()
policy = policy * valid_moves
policy = policy / np.sum(policy)

print("5x5棋盘位置编号 (action index):")
print("="*40)
for row in range(5):
    for col in range(5):
        idx = row * 5 + col
        print(f"{idx:2d}", end=" ")
    print()

print("\n策略概率分布 (reshape成5x5棋盘):")
print("="*40)
policy_board = policy.reshape(5, 5)
for row in range(5):
    for col in range(5):
        print(f"{policy_board[row, col]:.3f}", end=" ")
    print()

center_idx = 2 * 5 + 2
print(f"\n中心位置 (2,2) 的action index: {center_idx}")
print(f"中心位置的概率: {policy[center_idx]:.6f}")

print(f"\nTop 10 位置:")
top10 = np.argsort(policy)[-10:][::-1]
for i, idx in enumerate(top10):
    row, col = idx // 5, idx % 5
    print(f"{i+1}. 位置({row},{col}) [action {idx}]: {policy[idx]:.6f}")
