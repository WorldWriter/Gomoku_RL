"""
随机智能体实现
作为基线对比模型
"""

import numpy as np


class RandomAgent:
    """随机落子的智能体"""

    def __init__(self, board_size=15):
        """
        初始化随机智能体

        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size

    def get_action(self, state, valid_actions=None, training=False):
        """
        从合法动作中随机选择一个

        Args:
            state: 当前棋盘状态 (未使用，但保持接口一致)
            valid_actions: 合法动作列表，如果为 None 则从状态中计算
            training: 是否处于训练模式（随机智能体不使用此参数）

        Returns:
            选择的动作
        """
        if valid_actions is None:
            # 从状态中找出所有空位
            valid_positions = np.argwhere(state == 0)
            if len(valid_positions) == 0:
                return None
            # 随机选择一个空位
            row, col = valid_positions[np.random.randint(len(valid_positions))]
            return row * self.board_size + col
        else:
            # 从给定的合法动作中随机选择
            if len(valid_actions) == 0:
                return None
            return np.random.choice(valid_actions)

    def reset(self):
        """重置智能体（随机智能体无需重置）"""
        pass
