"""
Device utility for CUDA/CPU detection
Supports cross-platform training (Mac CPU / Windows GPU)
"""

import torch


def get_device(prefer_cuda=True):
    """
    Automatically detect and return the best available device

    Args:
        prefer_cuda: If True, use CUDA when available, otherwise use CPU

    Returns:
        torch.device: The device to use for training/inference
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        if prefer_cuda and not torch.cuda.is_available():
            print("⚠ CUDA not available, using CPU")
        else:
            print("✓ Using CPU")

    return device


def get_device_info():
    """
    Get detailed device information

    Returns:
        dict: Device information including CUDA availability, device name, etc.
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info['devices'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'name': torch.cuda.get_device_name(i),
                'total_memory': props.total_memory,
                'compute_capability': f"{props.major}.{props.minor}"
            })

    return info


def print_device_info():
    """
    Print detailed device information in a readable format
    """
    info = get_device_info()

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Count: {info['device_count']}")
        print()
        for i, device in enumerate(info['devices']):
            print(f"GPU {i}: {device['name']}")
            print(f"  Total Memory: {device['total_memory'] / 1e9:.2f} GB")
            print(f"  Compute Capability: {device['compute_capability']}")
    else:
        print("No CUDA devices available - training will use CPU")

    print("=" * 60)
