import sys

# 添加E盘安装路径到sys.path
sys.path.insert(0, 'E:\pytorch_install\Lib\site-packages')

print("Python路径:", sys.path)

# 尝试导入torch并测试CUDA
try:
    import torch
    print("成功导入PyTorch")
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA设备数量:", torch.cuda.device_count())
        print("当前设备:", torch.cuda.current_device())
        print("设备名称:", torch.cuda.get_device_name(0))
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()