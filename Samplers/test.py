import torchvision
print(torchvision.__version__)

# transforms.v2 был добавлен в torchvision версии 0.15.0 и выше.

try:
    import torchvision.transforms.v2 as T
    print("transforms.v2 доступен!")
except ImportError:
    print("transforms.v2 НЕ доступен в вашей версии torchvision")

import torchvision.transforms
print(dir(torchvision.transforms))
# Ищите 'v2' в списке

import torch

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")