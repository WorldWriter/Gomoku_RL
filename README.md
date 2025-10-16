# 五子棋强化学习项目 (Gomoku RL)

一个简洁明了的五子棋强化学习项目，使用 DQN（Deep Q-Network）算法训练智能体学习五子棋策略。非常适合学习强化学习的核心概念！

## 项目特点

- **清晰的代码结构**：模块化设计，易于理解和扩展
- **完整的强化学习流程**：环境、智能体、训练、评估一应俱全
- **详细的注释**：代码中包含丰富的中文注释，便于学习
- **可视化训练过程**：自动生成训练曲线图
- **人机对弈**：训练完成后可与AI下棋

## 项目结构

```
Gomoku_RL/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖
├── gomoku/                   # 五子棋游戏核心模块
│   ├── __init__.py
│   ├── board.py             # 棋盘逻辑（规则、胜负判断）
│   └── env.py               # Gymnasium环境封装
├── agents/                   # 强化学习智能体
│   ├── __init__.py
│   ├── random_agent.py      # 随机智能体（基线）
│   └── dqn_agent.py         # DQN深度Q网络智能体
├── models/                   # 保存训练的模型
├── train.py                  # 训练脚本（自我对弈）
├── evaluate.py              # 评估脚本
└── play.py                  # 人机对弈界面
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练智能体

训练 DQN 智能体（默认对手为随机智能体）：

```bash
python train.py --episodes 1000 --board-size 15
```

**训练参数说明：**
- `--episodes`: 训练轮数（默认 1000）
- `--device`: 计算设备选择，可选值为 `auto`（默认，自动检测）、`cpu`（使用CPU）或 `cuda`（使用GPU）

## GPU支持

### 检查CUDA支持

运行以下命令检查您的环境是否支持CUDA：

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 安装CUDA版本的PyTorch

如果您的系统有NVIDIA GPU并支持CUDA，请安装CUDA版本的PyTorch以获得更快的训练速度：

```bash
# 对于CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 对于CUDA 12.1
pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121
```

> 注意：请根据您的CUDA版本选择正确的安装命令。您也可以访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取适合您系统的最新安装命令。

### 使用GPU训练

安装好CUDA版本的PyTorch后，可以使用以下命令启用GPU训练：

```bash
python train.py --device cuda
```
- `--board-size`: 棋盘大小（默认 15）
- `--save-dir`: 模型保存目录（默认 models）
- `--save-interval`: 模型保存间隔（默认 100 轮）
- `--opponent`: 对手类型，`random`（随机）或 `self`（自我对弈）

训练过程中会自动：
- 每 100 轮打印训练进度
- 每 100 轮保存模型检查点
- 训练结束后生成训练曲线图 `training_curves.png`

### 3. 评估智能体

评估训练好的模型性能：

```bash
python evaluate.py --model-path models/dqn_agent_final.pth --num-games 100
```

**评估参数说明：**
- `--model-path`: 模型文件路径（必需）
- `--num-games`: 评估游戏数量（默认 100）
- `--board-size`: 棋盘大小（默认 15）
- `--opponent`: 对手类型（默认 random）
- `--verbose`: 打印详细信息

### 4. 人机对弈

与训练好的AI对弈：

```bash
python play.py --model-path models/dqn_agent_final.pth
```

**对弈参数说明：**
- `--model-path`: 模型文件路径（不指定则使用随机智能体）
- `--board-size`: 棋盘大小（默认 15）
- `--ai-first`: AI先手（默认人类先手）

**如何下棋：**
- 按照提示输入落子坐标，格式为：`行 列`（例如：`7 7`）
- 坐标范围：0 到 14（对于 15×15 棋盘）
- 黑子用 ● 表示，白子用 ○ 表示

## 强化学习核心概念

本项目涵盖了以下强化学习核心概念：

### 1. 马尔可夫决策过程（MDP）
- **状态（State）**：15×15 的棋盘状态
- **动作（Action）**：在棋盘某个位置落子
- **奖励（Reward）**：
  - 获胜：+1
  - 失败：-1
  - 平局：0
  - 非法落子：-10
  - 每步：-0.01（鼓励快速获胜）

### 2. Q-Learning 与 DQN
- **Q值**：在某个状态下采取某个动作的期望回报
- **Q网络**：使用卷积神经网络（CNN）近似Q函数
- **目标网络**：稳定训练过程，每隔一定步数更新

### 3. 经验回放（Experience Replay）
- 存储过往经验：(state, action, reward, next_state, done)
- 随机采样批量数据进行训练
- 打破数据相关性，提高训练稳定性

### 4. 探索-利用权衡（Exploration-Exploitation）
- **ε-greedy策略**：
  - 以 ε 概率随机探索
  - 以 1-ε 概率选择最优动作
  - ε 随训练逐渐衰减（从 1.0 到 0.01）

### 5. 自我对弈（Self-Play）
- 智能体与自己或其他智能体对弈
- 通过博弈学习策略
- 持续提升棋力

## 代码详解

### 五子棋环境 (gomoku/env.py)

环境遵循 Gymnasium 标准接口：

```python
env = GomokuEnv(board_size=15)

# 重置环境
state, info = env.reset()

# 执行动作
next_state, reward, done, truncated, info = env.step(action)
```

### DQN智能体 (agents/dqn_agent.py)

核心组件：

1. **Q网络**：3层卷积 + 2层全连接
2. **经验回放缓冲区**：容量 10000
3. **训练步骤**：
   - 从缓冲区采样
   - 计算目标Q值
   - 更新Q网络参数
   - 定期更新目标网络

### 训练流程 (train.py)

自我对弈训练循环：

```python
for episode in range(episodes):
    # 1. 自我对弈一局
    episode_data, winner = self_play_episode(agent1, agent2, env)

    # 2. 存储经验
    for state, action, reward, next_state, done in episode_data:
        agent.store_transition(state, action, reward, next_state, done)

    # 3. 训练网络
    loss = agent.train_step()

    # 4. 更新探索率
    agent.update_epsilon()
```

## 训练建议

1. **初期训练（前 500 轮）**：
   - 对手使用随机智能体
   - 智能体快速学习基本策略

2. **进阶训练（500-2000 轮）**：
   - 增加训练轮数
   - 可以尝试自我对弈（`--opponent self`）

3. **参数调优**：
   - 学习率：0.001（默认）
   - 折扣因子 γ：0.99（默认）
   - 批量大小：64（默认）
   - 探索率衰减：0.995（默认）

4. **训练技巧**：
   - 使用 GPU 加速训练（自动检测）
   - 观察训练曲线判断收敛情况
   - 定期评估模型性能

## 扩展方向

想要进一步提升项目？可以尝试：

1. **改进网络结构**：
   - 使用残差网络（ResNet）
   - 增加网络深度

2. **高级算法**：
   - Double DQN
   - Dueling DQN
   - Rainbow DQN
   - AlphaGo Zero 风格的 MCTS + 神经网络

3. **增强训练**：
   - 优先经验回放（Prioritized Experience Replay）
   - 课程学习（Curriculum Learning）
   - 对抗训练

4. **可视化改进**：
   - 图形界面（pygame/tkinter）
   - 实时训练监控（tensorboard）

## 学习资源

- **强化学习书籍**：
  - 《Reinforcement Learning: An Introduction》 - Sutton & Barto
  - 《深度强化学习》 - 王树森等

- **在线课程**：
  - CS285 (Berkeley)
  - DeepMind RL Course

- **论文**：
  - DQN: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
  - AlphaGo: "Mastering the game of Go with deep neural networks" (Silver et al., 2016)

## 常见问题

**Q: 训练多久才能打败随机智能体？**
A: 通常 500-1000 轮训练后，智能体就能稳定战胜随机智能体（胜率 > 80%）。

**Q: 为什么训练很慢？**
A: 15×15 棋盘的动作空间较大。可以尝试：
- 减小棋盘尺寸（如 9×9）
- 使用 GPU 训练
- 减少网络层数

**Q: 如何判断模型是否训练好？**
A: 观察指标：
- 对随机智能体胜率 > 80%
- 训练曲线趋于稳定
- 探索率降至较低水平

**Q: 可以用在其他棋类游戏吗？**
A: 可以！只需修改棋盘规则和胜负判断逻辑，框架可复用于：
- 井字棋（Tic-Tac-Toe）
- 四子棋（Connect Four）
- 围棋（Go）等

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

本项目旨在帮助初学者理解强化学习在棋类游戏中的应用。祝学习愉快！
