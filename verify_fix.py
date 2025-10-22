"""
验证修复是否正常工作
测试：
1. 配置文件是否正确加载
2. Dirichlet噪声是否可以正常工作
3. MCTS是否能正常运行
"""
import sys
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')
import torch
import numpy as np
from gomoku import GomokuGame
from alphazero import AlphaZeroNet, MCTS
from configs.config_5x5 import get_config

def verify_config():
    """验证配置文件修改"""
    print("="*60)
    print("1. 验证配置文件")
    print("="*60)

    config = get_config()

    checks = {
        'NUM_SIMULATIONS': (config['num_simulations'], 800, "MCTS模拟次数"),
        'C_PUCT': (config['c_puct'], 2.0, "探索常数"),
        'TEMPERATURE_THRESHOLD': (config['temperature_threshold'], 8, "温度阈值"),
        'DIRICHLET_ALPHA': (config.get('dirichlet_alpha'), 0.3, "Dirichlet alpha"),
        'DIRICHLET_EPSILON': (config.get('dirichlet_epsilon'), 0.25, "Dirichlet epsilon"),
    }

    all_pass = True
    for name, (actual, expected, desc) in checks.items():
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"{status} {desc}: {actual} (期望: {expected})")

    return all_pass

def verify_dirichlet_noise():
    """验证Dirichlet噪声功能"""
    print("\n" + "="*60)
    print("2. 验证Dirichlet噪声功能")
    print("="*60)

    config = get_config()
    game = GomokuGame(board_size=5)
    network = AlphaZeroNet(
        board_size=5,
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks']
    )
    network.eval()

    # 测试不带噪声的MCTS
    print("\n测试1: 不带Dirichlet噪声")
    mcts1 = MCTS(network, game, config)
    game.reset()
    policy1 = mcts1.get_action_probs(game, temperature=1.0, add_dirichlet_noise=False)
    print(f"  策略熵: {-np.sum(policy1 * np.log(policy1 + 1e-8)):.4f}")

    # 测试带噪声的MCTS
    print("\n测试2: 带Dirichlet噪声")
    mcts2 = MCTS(network, game, config)
    game.reset()
    policy2 = mcts2.get_action_probs(game, temperature=1.0, add_dirichlet_noise=True)
    print(f"  策略熵: {-np.sum(policy2 * np.log(policy2 + 1e-8)):.4f}")

    # 比较差异
    diff = np.abs(policy1 - policy2).sum()
    print(f"\n策略差异 (L1距离): {diff:.4f}")

    if diff > 0.1:
        print("✓ Dirichlet噪声正常工作 (策略有明显变化)")
        return True
    else:
        print("✗ Dirichlet噪声可能未生效 (策略变化太小)")
        return False

def verify_mcts_performance():
    """验证MCTS搜索性能"""
    print("\n" + "="*60)
    print("3. 验证MCTS搜索性能")
    print("="*60)

    config = get_config()
    game = GomokuGame(board_size=5)
    network = AlphaZeroNet(
        board_size=5,
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks']
    )
    network.eval()

    print(f"MCTS配置: {config['num_simulations']}次模拟, c_puct={config['c_puct']}")

    # 测试MCTS能否运行
    mcts = MCTS(network, game, config)
    game.reset()

    import time
    start = time.time()
    policy = mcts.get_action_probs(game, temperature=1.0, add_dirichlet_noise=True)
    elapsed = time.time() - start

    print(f"✓ MCTS运行成功")
    print(f"  耗时: {elapsed:.2f}秒")
    print(f"  策略熵: {-np.sum(policy * np.log(policy + 1e-8)):.4f}")
    print(f"  最大概率: {np.max(policy):.4f}")

    # 检查根节点访问次数
    if mcts.root is not None:
        total_visits = sum(child.visit_count for child in mcts.root.children.values())
        print(f"  根节点总访问次数: {total_visits}")

        # 第一次模拟用于初始化根节点，所以实际访问次数是 num_simulations - 1
        expected_visits = config['num_simulations'] - 1
        if total_visits == expected_visits:
            print(f"✓ 访问次数正确 (第一次模拟用于初始化根节点)")
            return True
        else:
            print(f"✗ 访问次数异常 (期望: {expected_visits})")
            return False

    return False

def main():
    """运行所有验证"""
    print("\n" + "="*60)
    print("AlphaZero 五子棋修复验证")
    print("="*60)

    results = {}

    try:
        results['config'] = verify_config()
        results['dirichlet'] = verify_dirichlet_noise()
        results['mcts'] = verify_mcts_performance()
    except Exception as e:
        print(f"\n✗ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)

    all_pass = all(results.values())

    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")

    if all_pass:
        print("\n✓ 所有验证通过！可以开始训练。")
        print("\n推荐训练命令:")
        print("  python train.py --board_size 5 --iterations 100")
    else:
        print("\n✗ 部分验证失败，请检查修复。")

    return all_pass

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
