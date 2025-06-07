import torch
import torchvision.transforms.v2 as T
from PIL import Image
import numpy as np

# Создаем тестовое изображение
test_image = Image.new('RGB', (100, 100), color='red')
# Или numpy array
# test_image = np.random.rand(100, 100, 3).astype(np.float32)

# Пробуем преобразование
try:
    tensor = T.ToTensor()(test_image)
    print(f"Успешно! Тип: {type(tensor)}, форма: {tensor.shape}")
except Exception as e:
    print(f"Ошибка: {e}")