"""
五子棋棋盘实现
包含棋盘状态管理、落子、胜负判断等核心逻辑
"""

import numpy as np


class GomokuBoard:
    """五子棋棋盘类"""

    def __init__(self, size=15):
        """
        初始化棋盘

        Args:
            size: 棋盘大小，默认15x15
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0: 空, 1: 黑子, -1: 白子
        self.current_player = 1  # 1: 黑子先手, -1: 白子
        self.last_move = None  # 记录最后一步落子位置

    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.last_move = None

    def get_valid_moves(self):
        """
        获取所有合法落子位置

        Returns:
            合法位置的列表 [(row, col), ...]
        """
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, row, col):
        """
        在指定位置落子

        Args:
            row: 行索引
            col: 列索引

        Returns:
            是否落子成功
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False

        if self.board[row, col] != 0:
            return False

        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.current_player = -self.current_player  # 切换玩家
        return True

    def check_winner(self):
        """
        检查是否有玩家获胜

        Returns:
            1: 黑子获胜
            -1: 白子获胜
            0: 游戏继续
        """
        if self.last_move is None:
            return 0

        row, col = self.last_move
        player = self.board[row, col]

        # 检查四个方向：横、竖、主对角线、副对角线
        directions = [
            (0, 1),   # 横向
            (1, 0),   # 纵向
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]

        for dr, dc in directions:
            count = 1  # 包含当前落子

            # 正方向检查
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # 反方向检查
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return player

        return 0

    def is_full(self):
        """检查棋盘是否已满"""
        return not np.any(self.board == 0)

    def is_game_over(self):
        """
        检查游戏是否结束

        Returns:
            (游戏是否结束, 获胜者)
        """
        winner = self.check_winner()
        if winner != 0:
            return True, winner

        if self.is_full():
            return True, 0  # 平局

        return False, 0

    def get_state(self):
        """
        获取当前棋盘状态

        Returns:
            棋盘状态的副本
        """
        return self.board.copy()

    def render(self):
        """打印棋盘到控制台"""
        print("\n  ", end="")
        for i in range(self.size):
            print(f"{i:2}", end=" ")
        print()

        for i in range(self.size):
            print(f"{i:2}", end=" ")
            for j in range(self.size):
                if self.board[i, j] == 1:
                    print(" ●", end=" ")  # 黑子
                elif self.board[i, j] == -1:
                    print(" ○", end=" ")  # 白子
                else:
                    print(" ·", end=" ")  # 空位
            print()
        print()
