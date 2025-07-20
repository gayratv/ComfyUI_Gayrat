import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import folder_paths


class ConcatImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "background_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat_images"
    CATEGORY = "image processing"

    def concat_images(self, image1, image2=None, image3=None, image4=None, background_color="#000000"):
        # Собираем все поданные изображения в один список
        images_in = [img for img in [image1, image2, image3, image4] if img is not None]
        num_images = len(images_in)

        if num_images == 0:
            # Если изображений нет (хотя image1 обязателен), возвращаем пустой тензор
            return (torch.zeros(1, 1, 1, 3),)

        # Конвертируем тензоры в PIL изображения
        images_pil = [Image.fromarray((img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)) for img in images_in]

        # Определяем максимальную ширину и высоту
        max_w = max(img.width for img in images_pil)
        max_h = max(img.height for img in images_pil)

        # --- Новая логика определения макета ---
        if num_images == 1:
            # 1 изображение: просто возвращаем его
            final_image = images_pil[0]
        elif num_images == 2:
            # 2 изображения: макет 1x2 (в ряд)
            final_image = Image.new('RGB', (max_w * 2, max_h), background_color)
            final_image.paste(images_pil[0], (0, 0))
            final_image.paste(images_pil[1], (max_w, 0))
        elif num_images == 3 or num_images == 4:
            # 3 или 4 изображения: макет 2x2 (сетка)
            final_image = Image.new('RGB', (max_w * 2, max_h * 2), background_color)
            final_image.paste(images_pil[0], (0, 0))
            if num_images > 1:
                final_image.paste(images_pil[1], (max_w, 0))
            if num_images > 2:
                final_image.paste(images_pil[2], (0, max_h))
            if num_images > 3:
                final_image.paste(images_pil[3], (max_w, max_h))

        # Конвертируем итоговое изображение обратно в тензор
        final_tensor = torch.from_numpy(np.array(final_image).astype(np.float32) / 255.0).unsqueeze(0)

        return (final_tensor,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Простая проверка на изменение для перезапуска
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "ConcatImages": ConcatImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatImages": "Concat Images Logic"
}