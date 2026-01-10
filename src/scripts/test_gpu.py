"""
GPU verification script.
Tests CUDA availability and GPU specifications.
"""

import torch
from src.core.utils import get_device, get_gpu_info

print("="*60)
print("GPU VERIFICATION TEST")
print("="*60)

print("\n1. PyTorch CUDA Availability:")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   - Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"   - Multi-Processor Count: {props.multi_processor_count}")
        print(f"   - Compute Capability: {props.major}.{props.minor}")
else:
    print("   No CUDA-capable GPU detected")
    print("   Possible reasons:")
    print("   - No NVIDIA GPU installed")
    print("   - CUDA drivers not installed")
    print("   - PyTorch CPU-only version installed")

print("\n2. Device Detection Function:")
device = get_device(prefer_gpu=True, verbose=True)

print("\n3. GPU Information:")
gpu_info = get_gpu_info()
if gpu_info:
    for key, value in gpu_info.items():
        print(f"   {key}: {value}")
else:
    print("   No GPU information available")

print("\n4. Test GPU Memory Allocation:")
if torch.cuda.is_available():
    try:
        # Allocate small tensor on GPU
        test_tensor = torch.randn(1000, 1000).to(device)
        print(f"   Successfully allocated tensor on {device}")
        print(f"   Tensor shape: {test_tensor.shape}")
        print(f"   GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"   GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        del test_tensor
        torch.cuda.empty_cache()
        print("   Memory cleared successfully")
    except Exception as e:
        print(f"   Error during GPU allocation: {e}")
else:
    print("   Skipped - no GPU available")

print("\n" + "="*60)
print("GPU VERIFICATION COMPLETE")
print("="*60)
