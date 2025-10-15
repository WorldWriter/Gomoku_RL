"""
评估脚本：测试训练好的 DQN 智能体
"""

import argparse
from tqdm import tqdm

from gomoku import GomokuEnv
from agents import DQNAgent, RandomAgent


def evaluate_agent(agent, opponent, env, num_games=100, verbose=False):
    """
    评估智能体性能

    Args:
        agent: 被评估的智能体
        opponent: 对手智能体
        env: 游戏环境
        num_games: 评估游戏数量
        verbose: 是否打印详细信息

    Returns:
        评估结果字典
    """
    wins = 0
    losses = 0
    draws = 0

    for game in tqdm(range(num_games), desc="评估进度"):
        state, _ = env.reset()
        done = False
        current_player = 1  # 1: agent, -1: opponent

        while not done:
            valid_actions = env.get_valid_actions()

            if current_player == 1:
                # Agent 的回合
                action = agent.get_action(state, valid_actions, training=False)
            else:
                # Opponent 的回合
                action = opponent.get_action(state, valid_actions)

            if action is None:
                break

            state, reward, done, truncated, info = env.step(action)

            if verbose and done:
                print(f"\n游戏 {game + 1} 结束")
                env.render()

            current_player = -current_player

        # 统计结果
        winner = info.get("winner", 0)
        if winner == 1:
            wins += 1
            if verbose:
                print("结果: Agent 获胜!")
        elif winner == -1:
            losses += 1
            if verbose:
                print("结果: Opponent 获胜!")
        else:
            draws += 1
            if verbose:
                print("结果: 平局!")

    # 计算统计信息
    win_rate = wins / num_games * 100
    loss_rate = losses / num_games * 100
    draw_rate = draws / num_games * 100

    results = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="评估训练好的五子棋 DQN 智能体")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型文件路径",
    )
    parser.add_argument(
        "--board-size", type=int, default=15, help="棋盘大小 (默认: 15)"
    )
    parser.add_argument(
        "--num-games", type=int, default=100, help="评估游戏数量 (默认: 100)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random"],
        help="对手类型 (默认: random)",
    )
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")

    args = parser.parse_args()

    # 创建环境
    env = GomokuEnv(board_size=args.board_size)

    # 加载智能体
    agent = DQNAgent(board_size=args.board_size)
    try:
        agent.load(args.model_path)
        print(f"模型已加载: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 创建对手
    if args.opponent == "random":
        opponent = RandomAgent(board_size=args.board_size)
        print("对手: 随机智能体")

    print(f"\n开始评估，共 {args.num_games} 局")
    print(f"棋盘大小: {args.board_size}x{args.board_size}")

    # 评估
    results = evaluate_agent(
        agent, opponent, env, num_games=args.num_games, verbose=args.verbose
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"总游戏数: {args.num_games}")
    print(f"胜场: {results['wins']} ({results['win_rate']:.2f}%)")
    print(f"负场: {results['losses']} ({results['loss_rate']:.2f}%)")
    print(f"平局: {results['draws']} ({results['draw_rate']:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
