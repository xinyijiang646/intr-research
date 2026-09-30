import sys
import torch
import torchvision

print("=== INTR Environment Check ===")

print("Python version:", sys.version)
print("Python executable:", sys.executable)

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

print("Selected device:", device)