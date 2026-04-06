"""
Check CUDA Availability and Provide GPU Configuration Recommendations
"""

import torch
import sys

def check_cuda_availability():
    """Check if CUDA is available and provide detailed information."""
    print("="*70)
    print("CUDA/GPU AVAILABILITY CHECK")
    print("="*70)
    
    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print("\n✅ CUDA IS AVAILABLE!")
        print("="*70)
        
        # CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        # GPU details
        for i in range(num_gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Current device
        print(f"\nCurrent Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Test tensor creation on GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print("\n✅ Successfully created tensor on GPU!")
            print(f"Test tensor device: {test_tensor.device}")
            del test_tensor
        except Exception as e:
            print(f"\n❌ Error creating tensor on GPU: {e}")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS FOR YOUR GNN SCRIPTS")
        print("="*70)
        print("\nTo enable GPU acceleration in your training scripts, add this code:")
        print("\n# At the beginning of your script (after imports)")
        print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("print(f'Using device: {device}')")
        print("\n# Move model to GPU")
        print("model = model.to(device)")
        print("\n# Move data to GPU")
        print("data = data.to(device)")
        print("# OR for individual tensors:")
        print("data.x = data.x.to(device)")
        print("data.edge_index = data.edge_index.to(device)")
        print("data.y = data.y.to(device)")
        
        print("\n" + "="*70)
        print("EXPECTED PERFORMANCE IMPROVEMENT")
        print("="*70)
        print("With GPU acceleration, you can expect:")
        print("  • 5-20x faster training (depending on model size)")
        print("  • Ability to use larger batch sizes")
        print("  • Faster hyperparameter tuning")
        print("  • More efficient for larger graphs")
        
    else:
        print("\n❌ CUDA IS NOT AVAILABLE")
        print("="*70)
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("  4. GPU drivers need updating")
        
        print("\n" + "="*70)
        print("HOW TO ENABLE CUDA")
        print("="*70)
        
        print("\n1. Check if you have an NVIDIA GPU:")
        print("   - Open Device Manager (Windows)")
        print("   - Look under 'Display adapters'")
        
        print("\n2. Install CUDA Toolkit:")
        print("   - Download from: https://developer.nvidia.com/cuda-downloads")
        print("   - Recommended: CUDA 11.8 or 12.1")
        
        print("\n3. Install PyTorch with CUDA support:")
        print("   - Visit: https://pytorch.org/get-started/locally/")
        print("   - For CUDA 11.8:")
        print("     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   - For CUDA 12.1:")
        print("     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n4. Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
        
        print("\n" + "="*70)
        print("CURRENT STATUS")
        print("="*70)
        print("Your GNN scripts are currently running on CPU.")
        print("This is fine for small graphs but GPU would be much faster.")
    
    # Check cuDNN
    if cuda_available:
        print("\n" + "="*70)
        print("cuDNN STATUS")
        print("="*70)
        cudnn_available = torch.backends.cudnn.enabled
        print(f"cuDNN Enabled: {cudnn_available}")
        if cudnn_available:
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print("✅ cuDNN is available for optimized convolution operations")
    
    return cuda_available


def benchmark_cpu_vs_gpu():
    """Simple benchmark to show CPU vs GPU performance difference."""
    if not torch.cuda.is_available():
        print("\n⚠️ Cannot run CPU vs GPU benchmark (no GPU available)")
        return
    
    print("\n" + "="*70)
    print("CPU vs GPU BENCHMARK")
    print("="*70)
    
    import time
    
    # Test matrix multiplication
    size = 5000
    iterations = 10
    
    print(f"\nBenchmarking matrix multiplication ({size}x{size}, {iterations} iterations)...")
    
    # CPU benchmark
    print("\nCPU:")
    cpu_times = []
    for i in range(iterations):
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        start = time.time()
        c = torch.mm(a, b)
        cpu_times.append(time.time() - start)
    cpu_avg = sum(cpu_times) / len(cpu_times)
    print(f"  Average time: {cpu_avg:.4f} seconds")
    
    # GPU benchmark
    print("\nGPU:")
    gpu_times = []
    for i in range(iterations):
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        torch.cuda.synchronize()
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - start)
    gpu_avg = sum(gpu_times) / len(gpu_times)
    print(f"  Average time: {gpu_avg:.4f} seconds")
    
    # Speedup
    speedup = cpu_avg / gpu_avg
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster than CPU")


if __name__ == "__main__":
    cuda_available = check_cuda_availability()
    
    if cuda_available:
        try:
            benchmark_cpu_vs_gpu()
        except Exception as e:
            print(f"\nBenchmark error: {e}")
    
    print("\n" + "="*70)
    print("✓ CHECK COMPLETE")
    print("="*70)
