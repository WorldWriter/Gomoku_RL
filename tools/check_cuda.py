import torch

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA是否可用
print(f"CUDA可用: {torch.cuda.is_available()}")

# 如果CUDA可用，显示CUDA版本
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到可用的CUDA设备")