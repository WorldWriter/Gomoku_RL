"""
简单测试：比较MCTS vs 纯网络策略
"""
import sys
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')
import torch
import numpy as np
from gomoku import GomokuGame
from alphazero import AlphaZeroNet, MCTS
from utils import get_device

def compare_mcts_vs_network():
    """比较MCTS输出和纯网络输出"""
    device = get_device()

    # 加载模型
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

    print("="*60)
    print("空棋盘: 纯网络策略 vs MCTS策略")
    print("="*60)

    # 1. 纯网络策略
    policy_net, value_net = network.predict(game.get_canonical_board())
    valid_moves = game.get_valid_moves()
    policy_net = policy_net * valid_moves
    policy_net = policy_net / np.sum(policy_net)

    print("\n【纯网络输出】")
    print(f"Value: {value_net:.4f}")
    print(f"策略熵: {-np.sum(policy_net * np.log(policy_net + 1e-8)):.4f}")
    print(f"策略标准差: {np.std(policy_net):.4f}")
    print("Top 5 moves:")
    top5_net = np.argsort(policy_net)[-5:][::-1]
    for idx in top5_net:
        row, col = idx // 5, idx % 5
        print(f"  ({row},{col}): {policy_net[idx]:.4f}")

    # 2. MCTS策略 (50次模拟)
    mcts_50 = MCTS(network, game, {'num_simulations': 50, 'c_puct': 1.0})
    policy_mcts_50 = mcts_50.get_action_probs(game, temperature=1.0)

    print("\n【MCTS 50次模拟】")
    print(f"策略熵: {-np.sum(policy_mcts_50 * np.log(policy_mcts_50 + 1e-8)):.4f}")
    print(f"策略标准差: {np.std(policy_mcts_50):.4f}")
    print("Top 5 moves:")
    top5_mcts50 = np.argsort(policy_mcts_50)[-5:][::-1]
    for idx in top5_mcts50:
        row, col = idx // 5, idx % 5
        print(f"  ({row},{col}): {policy_mcts_50[idx]:.4f}")

    # 3. MCTS策略 (200次模拟 - 训练时使用的)
    mcts_200 = MCTS(network, game, {'num_simulations': 200, 'c_puct': 1.0})
    policy_mcts_200 = mcts_200.get_action_probs(game, temperature=1.0)

    print("\n【MCTS 200次模拟（训练设置）】")
    print(f"策略熵: {-np.sum(policy_mcts_200 * np.log(policy_mcts_200 + 1e-8)):.4f}")
    print(f"策略标准差: {np.std(policy_mcts_200):.4f}")
    print("Top 5 moves:")
    top5_mcts200 = np.argsort(policy_mcts_200)[-5:][::-1]
    for idx in top5_mcts200:
        row, col = idx // 5, idx % 5
        print(f"  ({row},{col}): {policy_mcts_200[idx]:.4f}")

    # 4. 比较策略差异
    print("\n" + "="*60)
    print("策略对比分析")
    print("="*60)
    kl_net_mcts50 = np.sum(policy_net * np.log((policy_net + 1e-8) / (policy_mcts_50 + 1e-8)))
    kl_net_mcts200 = np.sum(policy_net * np.log((policy_net + 1e-8) / (policy_mcts_200 + 1e-8)))
    kl_mcts50_mcts200 = np.sum(policy_mcts_50 * np.log((policy_mcts_50 + 1e-8) / (policy_mcts_200 + 1e-8)))

    print(f"KL散度(网络 || MCTS-50):   {kl_net_mcts50:.4f}")
    print(f"KL散度(网络 || MCTS-200):  {kl_net_mcts200:.4f}")
    print(f"KL散度(MCTS-50 || MCTS-200): {kl_mcts50_mcts200:.4f}")

    if kl_net_mcts200 > 0.5:
        print("\n⚠️  警告: 网络策略和MCTS策略差异很大!")
        print("   这意味着网络的策略头没有学到有用的先验知识。")

    # 5. 测试Value头的准确性
    print("\n" + "="*60)
    print("Value预测准确性测试")
    print("="*60)

    # 随机下10局棋，看value预测和实际结果的相关性
    from alphazero.self_play import SelfPlay
    import random

    sp = SelfPlay(network, game, {'num_simulations': 100, 'c_puct': 1.0})

    value_errors = []
    for _ in range(5):
        # 玩一局棋
        train_examples = sp.play_game(temperature_schedule=lambda m: 1.0 if m < 10 else 0.0)

        # 随机选择3个中间位置
        sample_positions = random.sample(range(1, len(train_examples)-1), min(3, len(train_examples)-2))

        for pos in sample_positions:
            state, _, actual_value = train_examples[pos]
            predicted_policy, predicted_value = network.predict(state)

            error = abs(predicted_value - actual_value)
            value_errors.append(error)

    print(f"Value预测误差 (平均): {np.mean(value_errors):.4f}")
    print(f"Value预测误差 (标准差): {np.std(value_errors):.4f}")
    print(f"Value预测误差 (最大): {np.max(value_errors):.4f}")

    if np.mean(value_errors) > 0.5:
        print("\n⚠️  警告: Value预测误差很大!")
        print("   网络无法准确评估局面。")

if __name__ == '__main__':
    compare_mcts_vs_network()
