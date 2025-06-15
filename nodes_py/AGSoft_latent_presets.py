# https://t.me/prompt_by_art/1274
# НЕ рекомендовано

import torch
from comfy.sd import VAE
from nodes import EmptyLatentImage

class EmptyLatentImagePreset:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        presets = [
            "1536x640",
            "1344x768",
            "1216x832",
            "1152x896",
            "1024x768",
            "768x512",
            "512x512", 
            "640x640", 
            "768x768", 
            "1024x1024",
            "512x768",
            "768x1024",
            "896x1152",
            "832x1216",
            "768x1344",
            "640x1536",
            "Custom"
        ]
        return {
            "required": {
                "preset": (presets, {"default": "1024x1024"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "AGSoft/latent"

    def generate(self, preset, batch_size=1, width=None, height=None):
        # Получаем базовые размеры
        if preset != "Custom":
            dimensions = preset.split('x')
            width = int(dimensions[0])
            height = int(dimensions[1])

        # Корректируем размеры под требования модели
        adj_width = max(64, width // 8 * 8)
        adj_height = max(64, height // 8 * 8)

        # Создаем латент
        latent = torch.zeros([batch_size, 4, adj_height // 8, adj_width // 8])
        
        return ({"samples": latent}, adj_width, adj_height)

NODE_CLASS_MAPPINGS = {
    "EmptyLatentImagePreset": EmptyLatentImagePreset
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentImagePreset": "AGSoft Empty Latent Image (Preset)"
}