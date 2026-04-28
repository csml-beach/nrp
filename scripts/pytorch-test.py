import torch
import sys

def main():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Simple tensor operation on GPU
        x = torch.rand(5, 3).cuda()
        print("Successfully created a tensor on GPU:")
        print(x)
    else:
        print("CUDA is NOT available. Check your GPU configuration.")

if __name__ == "__main__":
    main()
