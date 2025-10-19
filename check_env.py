"""
Environment check script
Check PyTorch, CUDA availability, and test basic operations
Run this before training to verify your environment
"""

import sys
# 添加E盘PyTorch安装路径
sys.path.insert(0, 'E:\\pytorch_install\\Lib\\site-packages')
import torch
import numpy as np
from utils.device import print_device_info, get_device


def test_basic_operations():
    """Test basic PyTorch operations on the available device"""
    print("\nTesting basic operations...")
    print("-" * 60)

    try:
        device = get_device()

        # Test tensor creation
        print("✓ Tensor creation: ", end="")
        x = torch.randn(100, 100).to(device)
        print("OK")

        # Test matrix multiplication
        print("✓ Matrix multiplication: ", end="")
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        print("OK")

        # Test neural network operations
        print("✓ Neural network layers: ", end="")
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        input_tensor = torch.randn(1, 3, 15, 15).to(device)
        output = conv(input_tensor)
        print("OK")

        # Test backward pass
        print("✓ Backward pass: ", end="")
        loss = output.sum()
        loss.backward()
        print("OK")

        print("\nAll tests passed! ✓")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def check_python_packages():
    """Check if required packages are installed"""
    print("\nChecking Python packages...")
    print("-" * 60)

    required_packages = {
        'torch': torch,
        'numpy': np,
    }

    for name, module in required_packages.items():
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name:15s} {version}")

    # Check optional packages
    optional_packages = ['matplotlib', 'tqdm']
    for name in optional_packages:
        try:
            module = __import__(name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15s} {version}")
        except ImportError:
            print(f"✗ {name:15s} not installed (optional)")


def main():
    """Main function to run all checks"""
    print("=" * 60)
    print("AlphaZero Gomoku - Environment Check")
    print("=" * 60)

    # Print Python version
    print(f"\nPython Version: {sys.version}")

    # Check packages
    check_python_packages()

    # Print device info
    print()
    print_device_info()

    # Test operations
    success = test_basic_operations()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("Environment check completed successfully! ✓")
        print("You can now run training with: python train.py --board_size 5")
    else:
        print("Environment check failed. Please fix the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
