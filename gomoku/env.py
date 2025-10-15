"""
五子棋 Gymnasium 环境实现
符合 OpenAI Gym 接口标准
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .board import GomokuBoard


class GomokuEnv(gym.Env):
    """五子棋强化学习环境"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, board_size=15, opponent=None):
        """
        初始化环境

        Args:
            board_size: 棋盘大小
            opponent: 对手智能体，如果为 None 则需要外部提供对手动作
        """
        super().__init__()

        self.board_size = board_size
        self.board = GomokuBoard(board_size)
        self.opponent = opponent

        # 定义动作空间：选择棋盘上的一个位置 (0 到 board_size^2 - 1)
        self.action_space = spaces.Discrete(board_size * board_size)

        # 定义观察空间：棋盘状态 (board_size x board_size)
        # 值为 -1 (白子), 0 (空), 1 (黑子)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(board_size, board_size), dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        """
        重置环境

        Returns:
            observation: 初始观察状态
            info: 额外信息
        """
        super().reset(seed=seed)
        self.board.reset()

        observation = self.board.get_state()
        info = {"current_player": self.board.current_player}

        return observation, info

    def step(self, action):
        """
        执行一步动作

        Args:
            action: 动作 (0 到 board_size^2 - 1)

        Returns:
            observation: 新的观察状态
            reward: 奖励值
            terminated: 是否游戏结束
            truncated: 是否被截断
            info: 额外信息
        """
        # 将动作转换为棋盘坐标
        row = action // self.board_size
        col = action % self.board_size

        # 尝试落子
        if not self.board.make_move(row, col):
            # 非法落子，给予负奖励并结束游戏
            return (
                self.board.get_state(),
                -10,  # 非法落子的惩罚
                True,
                False,
                {"winner": -self.board.current_player, "invalid_move": True},
            )

        # 检查当前玩家是否获胜
        game_over, winner = self.board.is_game_over()

        if game_over:
            if winner == 1:  # 黑子（智能体）获胜
                reward = 1
            elif winner == -1:  # 白子（对手）获胜
                reward = -1
            else:  # 平局
                reward = 0

            return (
                self.board.get_state(),
                reward,
                True,
                False,
                {"winner": winner},
            )

        # 如果有对手，让对手落子
        if self.opponent is not None:
            opponent_action = self.opponent.get_action(self.board.get_state())
            opponent_row = opponent_action // self.board_size
            opponent_col = opponent_action % self.board_size

            if not self.board.make_move(opponent_row, opponent_col):
                # 对手非法落子，智能体获胜
                return (
                    self.board.get_state(),
                    1,
                    True,
                    False,
                    {"winner": 1, "opponent_invalid_move": True},
                )

            # 检查对手是否获胜
            game_over, winner = self.board.is_game_over()
            if game_over:
                if winner == -1:  # 对手获胜
                    reward = -1
                elif winner == 1:  # 不应该发生（智能体刚才没有获胜）
                    reward = 1
                else:  # 平局
                    reward = 0

                return (
                    self.board.get_state(),
                    reward,
                    True,
                    False,
                    {"winner": winner},
                )

        # 游戏继续，小的负奖励鼓励快速获胜
        reward = -0.01

        return (
            self.board.get_state(),
            reward,
            False,
            False,
            {"current_player": self.board.current_player},
        )

    def render(self):
        """渲染环境"""
        self.board.render()

    def get_valid_actions(self):
        """
        获取所有合法动作

        Returns:
            合法动作列表
        """
        valid_moves = self.board.get_valid_moves()
        return [row * self.board_size + col for row, col in valid_moves]

    def action_to_coords(self, action):
        """将动作转换为坐标"""
        return action // self.board_size, action % self.board_size

    def coords_to_action(self, row, col):
        """将坐标转换为动作"""
        return row * self.board_size + col
