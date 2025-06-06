import torch

print("=== GPU Test ===")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    print(f"✅ GPU is available")
    print(f" Device count: {device_count}")
    print(f" Using device: {device_name}")
else:
    print("❌ No GPU is available")

print("================")
