# AlphaZero 五子棋

一个实现 AlphaZero 算法用于五子棋（五子连珠）的强化学习项目。该项目支持多种棋盘尺寸（5×5、10×10、15×15）进行渐进式训练和学习。

## 功能特性

- **AlphaZero 算法**: 完整的 MCTS 和深度神经网络实现
- **多尺寸支持**: 可在 5×5、10×10 或 15×15 棋盘上训练
- **跨平台**: 支持 Mac (CPU) 和 Windows (CUDA GPU)
- **自我对弈训练**: 通过自我对弈生成训练数据
- **人机对战**: 与训练好的模型对战
- **模型评估**: 评估模型性能或与基准模型对比

## 项目结构

```
Gomoku_RL/
├── gomoku/              # 游戏逻辑
│   ├── board.py         # 棋盘实现
│   └── game.py          # 游戏包装器
├── alphazero/           # AlphaZero 实现
│   ├── network.py       # ResNet 策略-价值网络
│   ├── mcts.py          # 蒙特卡洛树搜索
│   ├── self_play.py     # 自我对弈数据生成
│   └── trainer.py       # 训练循环
├── configs/             # 配置文件
│   ├── config_5x5.py    # 5×5 棋盘配置
│   ├── config_10x10.py  # 10×10 棋盘配置
│   └── config_15x15.py  # 15×15 棋盘配置
├── utils/               # 工具函数
│   ├── device.py        # CUDA/CPU 检测
│   └── logger.py        # 日志工具
├── train.py             # 主训练脚本
├── play.py              # 人机对战界面
├── evaluate.py          # 模型评估
└── check_env.py         # 环境检查
```

## 安装

### 前置要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，Windows GPU 训练需要）

### 设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/Gomoku_RL.git
cd Gomoku_RL
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 检查环境：
```bash
python check_env.py
```

## 使用方法

### 1. 训练

在特定棋盘尺寸上训练模型：

```bash
# 在 5×5 上快速训练（用于测试）
python train.py --board_size 5

# 在 10×10 上中等训练
python train.py --board_size 10

# 在 15×15 上完整训练（需要 GPU）
python train.py --board_size 15
```

其他选项：
```bash
# 指定迭代次数
python train.py --board_size 5 --iterations 50

# 从检查点恢复训练
python train.py --board_size 10 --resume models/10x10/checkpoint_iter_50.pth

# 强制 CPU 训练
python train.py --board_size 5 --cpu
```

### 2. 与 AI 对战

与训练好的模型对战：

```bash
python play.py --checkpoint models/5x5/checkpoint_latest.pth --board_size 5

# 人类先手（黑子）
python play.py --checkpoint models/10x10/checkpoint_latest.pth --board_size 10 --human_first

# 调整 AI 强度（较少的模拟 = 更弱）
python play.py --checkpoint models/15x15/checkpoint_latest.pth --board_size 15 --simulations 200
```

### 3. 评估模型

评估模型对抗随机玩家：

```bash
python evaluate.py --checkpoint models/5x5/checkpoint_latest.pth --board_size 5 --num_games 100
```

比较两个模型：

```bash
python evaluate.py --checkpoint models/10x10/checkpoint_iter_100.pth \
                   --checkpoint2 models/10x10/checkpoint_iter_50.pth \
                   --baseline model --board_size 10 --num_games 50
```

## 训练策略

### 阶段 1: 5×5 棋盘（快速验证）
- **目的**: 验证算法正确性
- **时间**: CPU 上几个小时
- **配置**: 4 个 ResNet 块，200 次 MCTS 模拟
- **预期**: 智能体学习基本模式

### 阶段 2: 10×10 棋盘（中等规模）
- **目的**: 测试可扩展性
- **时间**: GPU 上 1-2 天
- **配置**: 8 个 ResNet 块，400 次 MCTS 模拟
- **预期**: 更复杂的策略

### 阶段 3: 15×15 棋盘（完整五子棋）
- **目的**: 实现强大的 AI 性能
- **时间**: GPU 上几天
- **配置**: 10 个 ResNet 块，800 次 MCTS 模拟
- **预期**: 接近专家级水平

## 配置

每种棋盘尺寸在 `configs/` 中都有自己的配置文件：

- `config_5x5.py`: 快速训练，适合 CPU
- `config_10x10.py`: 中等训练，推荐 GPU
- `config_15x15.py`: 完整训练，需要 GPU

可以调整的关键参数：
- `NUM_SIMULATIONS`: 每步的 MCTS 模拟次数
- `NUM_SELF_PLAY_GAMES`: 每次训练迭代的游戏数
- `BATCH_SIZE`: 训练批次大小
- `LEARNING_RATE`: 神经网络学习率
- `NUM_RES_BLOCKS`: 网络中的残差块数量

## 跨平台训练

### 在 Mac 上开发（CPU）
```bash
# 在 5×5 上快速迭代
python train.py --board_size 5 --iterations 20
```

### 在 Windows 上训练（GPU）
```bash
# 在较大棋盘上完整训练
python train.py --board_size 15 --iterations 500
```

代码会自动检测 CUDA 可用性，并在可用时使用 GPU。

## 算法细节

### AlphaZero 组件

1. **神经网络**: 具有两个头部的 ResNet 架构
   - 策略头部：输出移动概率
   - 价值头部：输出位置评估

2. **MCTS**: 具有 UCB 选择的蒙特卡洛树搜索
   - 上置信界公式：Q + c_puct × P × √N_parent / (1 + N)
   - 在根节点添加狄利克雷噪声进行探索

3. **自我对弈**: 生成训练数据
   - 使用当前网络 + MCTS 进行游戏
   - 存储（状态，策略，结果）元组
   - 使用对称性（旋转，翻转）进行数据增强

4. **训练**: 更新神经网络
   - 策略损失：MCTS 策略和网络策略之间的交叉熵
   - 价值损失：游戏结果和网络价值之间的 MSE
   - 使用 Adam 优化的组合损失

## 训练技巧

1. **从小开始**: 始终先在 5×5 上测试以确保一切正常
2. **监控损失**: 训练损失应该稳步下降
3. **保存检查点**: 每 N 次迭代保存模型
4. **GPU 内存**: 如果内存不足，调整批次大小
5. **训练时间**: 耐心等待 - 好的模型需要时间训练

## 故障排除

### CUDA 内存不足
- 减少配置中的 `BATCH_SIZE`
- 减少 MCTS 的 `NUM_SIMULATIONS`
- 使用更小的网络（更少的 `NUM_RES_BLOCKS`）

### 训练太慢
- 确保使用 CUDA（用 `python check_env.py` 检查）
- 减少 `NUM_SELF_PLAY_GAMES`
- 使用更小的棋盘尺寸进行测试

### 模型不学习
- 增加 `NUM_SIMULATIONS` 以获得更好的 MCTS 策略
- 检查损失曲线 - 它们应该下降
- 训练更多迭代
- 尝试不同的学习率

## 未来改进

- [ ] 使用多进程并行自我对弈
- [ ] 学习率调度
- [ ] 竞技场评估（新模型 vs 旧模型）
- [ ] 用于对战的 Web 界面
- [ ] 预训练模型下载
- [ ] 用于监控的 TensorBoard 集成

## 参考文献

- [AlphaGo Zero 论文](https://www.nature.com/articles/nature24270)
- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)
- [MCTS 综述](https://ieeexplore.ieee.org/document/6145622)
- [AlphaZero 五子棋](https://arxiv.org/abs/2309.01294?utm_source=chatgpt.com)
## 许可证

MIT 许可证 - 可自由将此项目用于学习和实验。

## 致谢

此项目是 AlphaZero 算法的学习实现，用于教育目的。