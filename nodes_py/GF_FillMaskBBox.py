import torch
import numpy as np
import cv2

# Нода GOS

class GF_FillMaskBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "extend_to_bottom": ("BOOLEAN", {"default": False}),
                "thickness": ("INT", {"default": 3, "min": 1, "max": 20}),
                "fill_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_mask", "image_with_fill", "image_with_bbox")
    FUNCTION = "process"
    CATEGORY = "Gayrat/Mask Processing"

    def process(self, image, mask, extend_to_bottom, thickness, fill_opacity):
        mask_np = mask.squeeze(0).cpu().numpy() * 255
        mask_np = mask_np.astype(np.uint8)
        
        image_np = image.squeeze(0).cpu().numpy() * 255
        image_np = image_np.astype(np.uint8).copy()
        h, w = image_np.shape[:2]

        bgr_color = (255, 0, 0)  # Синий цвет в BGR

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (mask, image, image)

        x, y, box_w, box_h = cv2.boundingRect(np.concatenate(contours))
        
        if extend_to_bottom:
            box_h = h - y

        # Создаем filled mask
        filled_mask = mask_np.copy()
        filled_mask[y:y+box_h, x:x+box_w] = 255

        # Создаем image_with_fill с заливкой
        overlay = image_np.copy()
        cv2.rectangle(overlay, (x, y), (x+box_w, y+box_h), bgr_color, -1)
        image_with_fill = cv2.addWeighted(overlay, fill_opacity, image_np, 1-fill_opacity, 0)

        # Создаем image_with_bbox ТОЛЬКО с рамкой (без заливки)
        image_with_bbox = image_np.copy()  # Используем оригинальное изображение
        cv2.rectangle(image_with_bbox, (x, y), (x+box_w, y+box_h), bgr_color, thickness)

        # Конвертируем результаты в тензоры
        filled_mask_tensor = torch.from_numpy(filled_mask / 255.0).unsqueeze(0)
        image_with_fill_tensor = torch.from_numpy(image_with_fill / 255.0).unsqueeze(0)
        image_with_bbox_tensor = torch.from_numpy(image_with_bbox / 255.0).unsqueeze(0)

        return (filled_mask_tensor, image_with_fill_tensor, image_with_bbox_tensor)

NODE_CLASS_MAPPINGS = {"GF_FillMaskBBox": GF_FillMaskBBox}
NODE_DISPLAY_NAME_MAPPINGS = {"GF_FillMaskBBox": "GF Fill BBox"}