# AlphaZero 五子棋修复说明

## ✅ 已修复的问题

### 1. **MCTS模拟次数不足** (config_5x5.py:15)
**问题**：5x5棋盘只使用200次MCTS模拟，无法有效纠正网络的错误先验概率
- **修复前**：`NUM_SIMULATIONS = 200`
- **修复后**：`NUM_SIMULATIONS = 800` （4倍提升）
- **影响**：MCTS搜索质量显著提升，能更好地探索棋局

### 2. **探索强度不足** (config_5x5.py:16)
**问题**：c_puct=1.0导致探索不足，MCTS过于依赖网络先验
- **修复前**：`C_PUCT = 1.0`
- **修复后**：`C_PUCT = 2.0`
- **影响**：增强探索，允许MCTS尝试网络给出低概率但可能有价值的走法

### 3. **温度调度不合理** (config_5x5.py:20)
**问题**：5x5棋盘平均对局10-15步，THRESHOLD=15过长导致整局都在探索
- **修复前**：`TEMPERATURE_THRESHOLD = 15`
- **修复后**：`TEMPERATURE_THRESHOLD = 8`
- **影响**：前8步探索（temperature=1），之后利用（temperature=0），更适合5x5棋盘

### 4. **缺少Dirichlet噪声机制** (config_5x5.py:22-24, mcts.py:178-194, self_play.py:58-59)
**问题**：无探索机制帮助发现被网络忽视的好位置
- **新增配置**：
  ```python
  DIRICHLET_ALPHA = 0.3      # Dirichlet分布参数
  DIRICHLET_EPSILON = 0.25   # 噪声混合比例
  ```
- **实现位置**：
  - `mcts.py:178-194` - 在根节点初始化时添加Dirichlet噪声
  - `self_play.py:58-59` - 在每局游戏第一步启用噪声
- **噪声公式**：`P_new = (1-ε)*P + ε*Dir(α)`
- **影响**：强制MCTS探索所有动作，即使网络给出极低概率

### 5. **人机对战AI先手bug** (play.py:155-160)
**问题**：AI先手时MCTS树不更新，导致搜索基于错误的棋盘状态
- **修复前**：只更新 `other_player` 的MCTS树
  ```python
  if isinstance(other_player, AIPlayer):
      other_player.mcts.update_root(action)
  ```
- **修复后**：同时更新 `current_player` 和 `other_player` 的MCTS树
  ```python
  if isinstance(current_player, AIPlayer):
      current_player.mcts.update_root(action)
  if isinstance(other_player, AIPlayer):
      other_player.mcts.update_root(action)
  ```
- **影响**：修复AI先手时的异常行为

---

## 🔍 根本原因分析

### 训练失败的恶性循环
```
随机初始化网络 → 给出错误先验（边缘96%概率）
                ↓
        MCTS搜索不足（200次）无法纠正
                ↓
        生成错误的训练数据
                ↓
        网络学习错误策略（Loss降低但策略错误）
                ↓
        更错误的先验 → 恶性循环
```

### 诊断证据
运行 `check_center.py` 发现：
- 空棋盘中心位置(2,2)概率仅 **0.5%**（应该最高）
- 4个边缘位置总概率 **96%**（完全错误）
- 策略呈现异常的"十字形"分布

运行 `simple_test.py` 发现：
- KL散度(网络||MCTS) = **0.13**（太小，说明MCTS几乎等于网络输出）
- MCTS未能纠正网络错误
- Value预测误差平均 **0.12**（可接受，但策略头已失效）

---

## 📊 修复效果预期

| 指标 | 修复前 | 修复后（预期） |
|------|--------|---------------|
| MCTS模拟次数 | 200 | 800 |
| 探索强度(c_puct) | 1.0 | 2.0 |
| 中心位置概率 | 0.5% | >15% |
| vs Random胜率 | 50% | >70% |
| 策略KL散度 | 0.13 | >0.5 |
| 训练时间/轮 | ~15s | ~60s (+4倍) |

---

## 🚀 使用指南

### 验证修复
```bash
python verify_fix.py
```
应该看到：
```
✓ 通过: config
✓ 通过: dirichlet
✓ 通过: mcts
✓ 所有验证通过！可以开始训练。
```

### 重新训练
```bash
# 基础训练（100轮，约100分钟）
python train.py --board_size 5 --iterations 100

# Windows上使用CUDA加速（推荐）
python train.py --board_size 5 --iterations 100

# 如果需要继续训练
python train.py --board_size 5 --iterations 200 --resume models/5x5/checkpoint_latest.pth
```

### 评估模型
```bash
# 对战随机玩家（50局）
python evaluate.py --checkpoint models/5x5/checkpoint_latest.pth \
                   --baseline random --board_size 5 \
                   --num_games 50 --simulations 800

# 人机对战（AI后手）
python play.py --checkpoint models/5x5/checkpoint_latest.pth \
               --board_size 5 --simulations 800 --human_first

# 人机对战（AI先手，现已修复bug）
python play.py --checkpoint models/5x5/checkpoint_latest.pth \
               --board_size 5 --simulations 800
```

### 检查训练质量
```bash
# 测试策略分布是否合理
python check_center.py

# 对比网络vs MCTS策略
python simple_test.py
```

**期望输出**：
- 中心位置概率 >10%
- 策略分布较均匀，不应集中在边缘
- KL散度 >0.3（说明MCTS在纠正网络）

---

## 📝 训练日志监控

训练过程中查看：
```bash
# 查看最新训练曲线
open logs/training_curve_latest.png

# 查看训练日志
cat logs/training_log.json | tail -20
```

关键指标：
- **Loss降低**：正常应该从~3降到~0.7
- **Buffer size**：应该增长到MAX_HISTORY_LEN (50,000)
- **时间/轮**：CPU约60秒，CUDA约10-15秒

---

## ⚠️ 注意事项

1. **训练时间**：CPU训练很慢，强烈建议使用CUDA
   - CPU: ~60秒/轮 × 100轮 = 100分钟
   - CUDA: ~10秒/轮 × 100轮 = 17分钟

2. **内存使用**：5x5棋盘内存占用约2GB，可接受

3. **何时停止训练**：
   - Loss稳定在0.7-0.8
   - vs Random胜率 >80%
   - 策略分布合理（中心位置有高概率）

4. **如果训练仍然失败**：
   - 检查 `simple_test.py` 输出
   - 确认KL散度 >0.3
   - 可能需要调整学习率或增加模拟次数

---

## 🛠️ 技术细节

### Dirichlet噪声工作原理
在self-play的每局游戏第一步：
```python
# 1. 网络给出先验概率 P
policy = network.predict(board)

# 2. 生成Dirichlet噪声
noise = np.random.dirichlet([0.3] * num_actions)

# 3. 混合：75%网络 + 25%噪声
policy_new = 0.75 * policy + 0.25 * noise

# 4. MCTS基于混合后的先验搜索
mcts.search(policy_new)
```

**作用**：即使网络给某个位置0.001%概率，噪声也会强制MCTS尝试，从而发现真正好的走法。

### UCB公式
```
UCB(action) = Q(action) + c_puct * P(action) * √N_parent / (1 + N(action))
```
- `c_puct=2.0` 增加探索项权重
- 当网络P很小时，仍有机会被探索（如果Q值高）

---

## 📚 参考文献
- AlphaGo Zero论文：c_puct典型值2-5，Dirichlet(α=0.3)
- 5x5棋盘特点：状态空间小，需要更多探索避免过拟合

---

## ✨ 总结
修复核心思路：**增强MCTS探索能力，降低对错误网络先验的依赖**

通过4倍模拟次数 + 2倍探索强度 + Dirichlet噪声的组合，MCTS现在能够：
1. 有足够的时间探索所有位置
2. 不被网络错误先验主导
3. 发现并学习正确的策略

预期修复后，训练将收敛到合理策略，胜率显著提升！
