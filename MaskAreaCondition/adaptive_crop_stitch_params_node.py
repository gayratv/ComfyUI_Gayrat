import torch

class AdaptiveCropStitchParams_Simplified_WithPassthrough: # Изменено имя класса
    """
    A node to dynamically generate context_factor, expand_px, and blend_px
    based on the mask area percentage relative to the image,
    and pass through the original image and mask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = (
        "IMAGE",  # Original image
        "MASK",   # Original mask
        "FLOAT",  # context_from_mask_extend_factor
        "INT",    # mask_expand_pixels
        "INT",    # mask_blend_pixels
    )
    RETURN_NAMES = (
        "image_out", # Имя для выходного изображения
        "mask_out",  # Имя для выходной маски
        "context_from_mask_extend_factor",
        "mask_expand_pixels",
        "mask_blend_pixels",
    )

    FUNCTION = "get_params"
    CATEGORY = "Gayrat_Simplified" # Категорию можно изменить или оставить

    def get_params(self, image: torch.Tensor, mask: torch.Tensor):

        # === Calculate mask area percentage ===
        if image is None or mask is None: # Добавлена проверка на None для image и mask
            # Если image или mask отсутствуют, возвращаем их как есть (None)
            # и значения по умолчанию для вычисляемых параметров.
            calculated_mask_area_percentage = 0.0
            # Возвращаем None для image и mask, если они были None на входе,
            # и дефолтные значения для остальных параметров, чтобы избежать ошибок ниже.
            if image is None: # Явно передаем None, если image is None
                 pass # image останется None
            if mask is None: # Явно передаем None, если mask is None
                 pass # mask останется None
        else:
            # Get dimensions of the first image in the batch
            # ComfyUI IMAGE tensor typically NHWC, H=shape[1], W=shape[2]
            if image.dim() == 4 and image.shape[3] in [1, 3, 4]:  # NHWC
                img_height, img_width = image.shape[1], image.shape[2]
            elif image.dim() == 3:  # Assuming CHW
                img_height, img_width = image.shape[1], image.shape[2]
            else:
                print(f"Warning: Unexpected image dimensions {image.shape}, attempting to get H, W.")
                img_height, img_width = image.shape[-2], image.shape[-1]

            img_area = img_height * img_width

            # Get the first mask in the batch
            # ComfyUI MASK tensor is typically (Batch, Height, Width)
            first_mask = mask[0]  # Shape: (Height, Width)

            # Sum of mask pixels (where mask > 0)
            active_mask_pixels = torch.sum(first_mask > 0).item()

            if img_area == 0:
                calculated_mask_area_percentage = 0.0
            else:
                calculated_mask_area_percentage = (active_mask_pixels / img_area) * 100.0

        # === Dynamic values based on the *calculated* mask_area_percentage ===
        # Эти значения будут вычислены даже если image/mask is None,
        # так как calculated_mask_area_percentage будет 0.0
        if calculated_mask_area_percentage < 10.0:  # Small mask
            context_factor_val = 2.0
            expand_px_val = 6
            blend_px_val = 12
        elif calculated_mask_area_percentage < 40.0:  # Medium mask
            context_factor_val = 1.5
            expand_px_val = 12
            blend_px_val = 24
        else:  # Large mask
            context_factor_val = 1.1
            expand_px_val = 24
            blend_px_val = 48

        # Передаем оригинальные image и mask на выход
        return (
            image, # Оригинальное изображение
            mask,  # Оригинальная маска
            context_factor_val,
            expand_px_val,
            blend_px_val,
        )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AdaptiveCropStitchParams_Simplified_WithPassthrough": AdaptiveCropStitchParams_Simplified_WithPassthrough
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveCropStitchParams_Simplified_WithPassthrough": "Simplified Adaptive Params + Passthrough"
}