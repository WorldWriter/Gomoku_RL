"""
DQN 智能体实现
使用深度Q网络学习五子棋策略
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Q网络：使用卷积神经网络处理棋盘状态"""

    def __init__(self, board_size=15):
        super(QNetwork, self).__init__()

        self.board_size = board_size

        # 卷积层：提取棋盘特征
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入状态 (batch_size, board_size, board_size)

        Returns:
            Q值 (batch_size, action_space_size)
        """
        # 添加通道维度
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 1, H, W)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # 展平
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(
        self,
        board_size=15,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        device=None,
    ):
        """
        初始化DQN智能体

        Args:
            board_size: 棋盘大小
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
            buffer_capacity: 经验回放缓冲区容量
            batch_size: 批量大小
            target_update_freq: 目标网络更新频率
            device: 计算设备
        """
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 设置计算设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 创建Q网络和目标网络
        self.q_network = QNetwork(board_size).to(self.device)
        self.target_network = QNetwork(board_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 训练计数
        self.update_count = 0

    def get_action(self, state, valid_actions=None, training=True):
        """
        选择动作（ε-greedy策略）

        Args:
            state: 当前状态
            valid_actions: 合法动作列表
            training: 是否处于训练模式

        Returns:
            选择的动作
        """
        # 探索：随机选择动作
        if training and random.random() < self.epsilon:
            if valid_actions is None:
                valid_positions = np.argwhere(state == 0)
                if len(valid_positions) == 0:
                    return None
                row, col = valid_positions[np.random.randint(len(valid_positions))]
                return row * self.board_size + col
            else:
                if len(valid_actions) == 0:
                    return None
                return random.choice(valid_actions)

        # 利用：选择Q值最大的动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # 如果提供了合法动作列表，只考虑合法动作
        if valid_actions is not None:
            if len(valid_actions) == 0:
                return None
            # 将非法动作的Q值设为极小值
            mask = np.full(self.board_size * self.board_size, -np.inf)
            mask[valid_actions] = q_values[valid_actions]
            q_values = mask

        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """执行一步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """保存模型"""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]

    def reset(self):
        """重置智能体（DQN不需要重置状态）"""
        pass
