"""
训练脚本：使用自我对弈训练 DQN 智能体
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加E盘PyTorch安装路径
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')

from gomoku import GomokuEnv
from agents import DQNAgent, RandomAgent


def self_play_episode(agent1, agent2, env):
    """
    自我对弈一局游戏

    Args:
        agent1: 黑子智能体
        agent2: 白子智能体
        env: 游戏环境

    Returns:
        episode_data: 游戏过程数据 [(state, action, reward, next_state, done), ...]
        winner: 获胜者 (1: agent1, -1: agent2, 0: 平局)
    """
    state, _ = env.reset()
    episode_data = []
    done = False
    current_agent = agent1
    other_agent = agent2
    player = 1  # 1: agent1, -1: agent2

    while not done:
        # 获取合法动作
        valid_actions = env.get_valid_actions()

        # 当前智能体选择动作
        action = current_agent.get_action(state, valid_actions, training=True)

        if action is None:
            break

        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)

        # 调整奖励（从当前玩家视角）
        adjusted_reward = reward * player

        # 存储经验
        episode_data.append((state, action, adjusted_reward, next_state, done, player))

        state = next_state

        # 切换玩家
        current_agent, other_agent = other_agent, current_agent
        player = -player

    # 获取获胜者
    winner = info.get("winner", 0)

    return episode_data, winner


def train(
    episodes=1000,
    board_size=15,
    save_dir="models",
    save_interval=100,
    opponent_type="random",
    device=None,
):
    """
    训练DQN智能体

    Args:
        episodes: 训练轮数
        board_size: 棋盘大小
        save_dir: 模型保存目录
        save_interval: 模型保存间隔
        opponent_type: 对手类型 ("random" 或 "self")
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建环境和智能体
    env = GomokuEnv(board_size=board_size)
    agent = DQNAgent(board_size=board_size, device=device)

    # 创建对手
    if opponent_type == "random":
        opponent = RandomAgent(board_size=board_size)
        print("对手：随机智能体")
    else:
        opponent = DQNAgent(board_size=board_size)
        print("对手：自我对弈")

    # 训练统计
    win_counts = []  # 胜场数
    loss_counts = []  # 负场数
    draw_counts = []  # 平局数
    losses = []  # 训练损失
    epsilons = []  # 探索率

    total_wins = 0
    total_losses = 0
    total_draws = 0

    print(f"开始训练，共 {episodes} 局")
    print(f"棋盘大小: {board_size}x{board_size}")
    print(f"设备: {agent.device}")

    for episode in tqdm(range(episodes), desc="训练进度"):
        # 自我对弈
        episode_data, winner = self_play_episode(agent, opponent, env)

        # 统计胜负
        if winner == 1:
            total_wins += 1
        elif winner == -1:
            total_losses += 1
        else:
            total_draws += 1

        # 存储经验并训练
        for state, action, reward, next_state, done, player in episode_data:
            # 只训练agent1（黑子）
            if player == 1:
                agent.store_transition(state, action, reward, next_state, done)

                # 训练
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        # 更新探索率
        agent.update_epsilon()

        # 记录统计信息
        if (episode + 1) % 10 == 0:
            win_counts.append(total_wins)
            loss_counts.append(total_losses)
            draw_counts.append(total_draws)
            epsilons.append(agent.epsilon)

        # 打印进度
        if (episode + 1) % 100 == 0:
            win_rate = total_wins / (episode + 1) * 100
            print(
                f"\n轮次 {episode + 1}/{episodes} - "
                f"胜率: {win_rate:.2f}% "
                f"(胜: {total_wins}, 负: {total_losses}, 平: {total_draws}) - "
                f"ε: {agent.epsilon:.3f}"
            )

        # 保存模型
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(save_dir, f"dqn_agent_{episode + 1}.pth")
            agent.save(model_path)
            print(f"模型已保存: {model_path}")

    # 保存最终模型
    final_model_path = os.path.join(save_dir, "dqn_agent_final.pth")
    agent.save(final_model_path)
    print(f"\n最终模型已保存: {final_model_path}")

    # 绘制训练曲线
    plot_training_curves(win_counts, loss_counts, draw_counts, losses, epsilons)

    print("\n训练完成！")
    print(f"总胜场: {total_wins}/{episodes} ({total_wins/episodes*100:.2f}%)")
    print(f"总负场: {total_losses}/{episodes} ({total_losses/episodes*100:.2f}%)")
    print(f"总平局: {total_draws}/{episodes} ({total_draws/episodes*100:.2f}%)")


def plot_training_curves(win_counts, loss_counts, draw_counts, losses, epsilons):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 胜负平统计
    episodes_x = np.arange(1, len(win_counts) + 1) * 10
    axes[0, 0].plot(episodes_x, win_counts, label="Wins", color="green")
    axes[0, 0].plot(episodes_x, loss_counts, label="Losses", color="red")
    axes[0, 0].plot(episodes_x, draw_counts, label="Draws", color="blue")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Win/Loss/Draw Statistics")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 胜率曲线
    total_games = np.array(win_counts) + np.array(loss_counts) + np.array(draw_counts)
    win_rate = np.array(win_counts) / total_games * 100
    axes[0, 1].plot(episodes_x, win_rate, color="purple")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Win Rate (%)")
    axes[0, 1].set_title("Win Rate Over Time")
    axes[0, 1].grid(True)

    # 训练损失
    if losses:
        axes[1, 0].plot(losses, color="orange", alpha=0.6)
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Training Loss")
        axes[1, 0].grid(True)

    # 探索率
    axes[1, 1].plot(episodes_x, epsilons, color="brown")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].set_title("Exploration Rate (ε)")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("训练曲线已保存: training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练五子棋 DQN 智能体")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="训练轮数 (默认: 1000)"
    )
    parser.add_argument(
        "--board-size", type=int, default=15, help="棋盘大小 (默认: 15)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models", help="模型保存目录 (默认: models)"
    )
    parser.add_argument(
        "--save-interval", type=int, default=100, help="模型保存间隔 (默认: 100)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "self"],
        help="对手类型 (默认: random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备 (默认: auto)"
    )

    args = parser.parse_args()
    
    # 确定设备
    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")
    if device.type == "cpu" and torch.cuda.is_available():
        print("警告: CUDA 可用但选择了 CPU 设备。使用 --device=cuda 启用 GPU 加速。")
    elif device.type == "cuda" and not torch.cuda.is_available():
        print("错误: CUDA 不可用，将使用 CPU 设备。")
        device = torch.device("cpu")

    train(
        episodes=args.episodes,
        board_size=args.board_size,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        opponent_type=args.opponent,
        device=device  # 传递设备参数
    )
