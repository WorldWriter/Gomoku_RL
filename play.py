"""
人机对弈界面：与训练好的 DQN 智能体下棋
"""

import sys
import argparse

# 添加E盘PyTorch安装路径
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')

import torch
from gomoku import GomokuBoard
from agents import DQNAgent, RandomAgent


def get_human_move(board):
    """
    获取人类玩家的落子位置

    Args:
        board: 棋盘对象

    Returns:
        (row, col) 坐标
    """
    while True:
        try:
            move = input("\n请输入落子位置 (格式: 行 列，例如: 7 7): ")
            parts = move.strip().split()

            if len(parts) != 2:
                print("输入格式错误！请输入两个数字，用空格分隔。")
                continue

            row, col = int(parts[0]), int(parts[1])

            if not (0 <= row < board.size and 0 <= col < board.size):
                print(f"位置超出范围！请输入 0-{board.size-1} 之间的数字。")
                continue

            if board.board[row, col] != 0:
                print("该位置已有棋子！请选择其他位置。")
                continue

            return row, col

        except ValueError:
            print("输入错误！请输入数字。")
        except KeyboardInterrupt:
            print("\n游戏已退出。")
            exit(0)


def play_game(agent, board_size=15, human_first=True):
    """
    人机对弈

    Args:
        agent: AI智能体
        board_size: 棋盘大小
        human_first: 人类是否先手
    """
    board = GomokuBoard(board_size)

    print("\n" + "=" * 50)
    print("五子棋人机对弈")
    print("=" * 50)
    print(f"棋盘大小: {board_size}x{board_size}")
    print(f"你执: {'黑子 (●)' if human_first else '白子 (○)'}")
    print(f"AI执: {'白子 (○)' if human_first else '黑子 (●)'}")
    print("=" * 50)

    # 渲染初始棋盘
    board.render()

    human_player = 1 if human_first else -1
    ai_player = -human_player

    while True:
        current_player = board.current_player

        if current_player == human_player:
            # 人类玩家回合
            print(f"\n你的回合 ({'黑子 ●' if human_player == 1 else '白子 ○'})")
            row, col = get_human_move(board)

        else:
            # AI回合
            print(f"\nAI思考中 ({'黑子 ●' if ai_player == 1 else '白子 ○'})...")

            # 获取合法动作
            valid_moves = board.get_valid_moves()
            valid_actions = [r * board_size + c for r, c in valid_moves]

            # AI选择动作
            action = agent.get_action(board.get_state(), valid_actions, training=False)

            if action is None:
                print("AI无法落子！")
                break

            row = action // board_size
            col = action % board_size

            print(f"AI 落子: ({row}, {col})")

        # 执行落子
        if not board.make_move(row, col):
            print("非法落子！")
            continue

        # 渲染棋盘
        board.render()

        # 检查游戏是否结束
        game_over, winner = board.is_game_over()

        if game_over:
            print("\n" + "=" * 50)
            if winner == human_player:
                print("🎉 恭喜你获胜！")
            elif winner == ai_player:
                print("😔 AI 获胜！")
            else:
                print("🤝 平局！")
            print("=" * 50)
            break


def main():
    parser = argparse.ArgumentParser(description="与训练好的五子棋 DQN 智能体对弈")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型文件路径（如果不指定则使用随机智能体）",
    )
    parser.add_argument(
        "--board-size", type=int, default=15, help="棋盘大小 (默认: 15)"
    )
    parser.add_argument(
        "--ai-first", action="store_true", help="AI先手（默认人类先手）"
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

    # 创建智能体
    if args.model_path:
        agent = DQNAgent(board_size=args.board_size, device=device)
        try:
            agent.load(args.model_path)
            print(f"模型已加载: {args.model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用随机智能体代替。")
            agent = RandomAgent(board_size=args.board_size)
    else:
        print("未指定模型路径，使用随机智能体。")
        agent = RandomAgent(board_size=args.board_size)

    # 开始游戏
    while True:
        play_game(agent, board_size=args.board_size, human_first=not args.ai_first)

        # 询问是否再来一局
        play_again = input("\n是否再来一局？(y/n): ")
        if play_again.lower() != "y":
            print("感谢游戏！再见！")
            break


if __name__ == "__main__":
    main()
