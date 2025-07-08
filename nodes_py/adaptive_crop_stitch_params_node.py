'''
Этот узел, названный в интерфейсе "Adaptive Params (Model Choice + Passthrough)", является вспомогательным. Он не изменяет изображение или маску, а вместо этого динамически генерирует четыре числовых параметра, которые можно передать в другие узлы (например, в VAEEncodeForInpaint). Это избавляет от необходимости настраивать их вручную.

Логика работы
Узел выполняет две основные задачи:

Анализ размера маски:

Он вычисляет общую площадь изображения.

Он считает количество пикселей в "активной" части маски (области, которую нужно изменить).

Затем он находит процентное соотношение площади маски к площади всего изображения.

Генерация параметров:

В зависимости от процента маски, он устанавливает три параметра, которые обычно используются для inpainting:

Маленькая маска (< 10%): Контекст (context_factor) делается большим (2.0), а расширение (expand_px) и сглаживание (blend_px) — маленькими. Это помогает модели лучше "понять", что находится вокруг небольшой области.

Средняя маска (10-40%): Устанавливаются средние значения.

Большая маска (> 40%): Контекст делается минимальным (1.1), так как большая часть изображения и так будет изменена. Расширение и сглаживание увеличиваются для более плавного перехода по краям.

В зависимости от выбора model_type, он устанавливает параметр target_size (целевой размер):

Если выбрана модель "Flux", target_size устанавливается на 1024.

Если выбрана модель "SD 1.5", target_size устанавливается на 512.

Это логично, так как модели Flux обычно работают с более высоким разрешением (1024x1024), а SD 1.5 была обучена на изображениях 512x512.

Выходные параметры (Outputs)
image_out: Исходное изображение (без изменений).

mask_out: Исходная маска (без изменений).

context_from_mask_extend_factor: Динамически рассчитанный коэффициент контекста (например, 2.0, 1.5 или 1.1).

mask_expand_pixels: Динамически рассчитанное расширение маски в пикселях (например, 6, 12 или 24).

mask_blend_pixels: Динамически рассчитанное сглаживание маски в пикселях (например, 12, 24 или 48).

target_size_out: Целевой размер, основанный на модели (1024 или 512).

В итоге, этот узел — удобный инструмент автоматизации, который делает рабочий процесс более гибким и быстрым, адаптируя ключевые параметры под конкретную задачу и модель.
'''

import torch

MODEL_TYPES = ["Flux", "SD 1.5"] # Список для выбора модели

class AdaptiveParamsWithModelChoice: # Изменено имя класса
    """
    A node to dynamically generate context_factor, expand_px, blend_px,
    and target_size based on mask area and selected model type,
    and pass through the original image and mask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model_type": (MODEL_TYPES, {"default": MODEL_TYPES[0]}), # Новый входной параметр
            }
        }

    RETURN_TYPES = (
        "IMAGE",  # Original image
        "MASK",   # Original mask
        "FLOAT",  # context_from_mask_extend_factor
        "INT",    # mask_expand_pixels
        "INT",    # mask_blend_pixels
        "INT",    # target_size
    )
    RETURN_NAMES = (
        "image_out",
        "mask_out",
        "context_from_mask_extend_factor",
        "mask_expand_pixels",
        "mask_blend_pixels",
        "target_size_out", # Имя для нового выходного параметра
    )

    FUNCTION = "get_params"
    CATEGORY = "Gayrat" # Категорию можно изменить

    def get_params(self, image: torch.Tensor, mask: torch.Tensor, model_type: str):

        # === Calculate mask area percentage ===
        if image is None or mask is None:
            calculated_mask_area_percentage = 0.0
        else:
            if image.dim() == 4 and image.shape[3] in [1, 3, 4]:  # NHWC
                img_height, img_width = image.shape[1], image.shape[2]
            elif image.dim() == 3:  # Assuming CHW
                img_height, img_width = image.shape[1], image.shape[2]
            else:
                print(f"Warning: Unexpected image dimensions {image.shape}, attempting to get H, W.")
                img_height, img_width = image.shape[-2], image.shape[-1]

            img_area = img_height * img_width
            first_mask = mask[0]
            active_mask_pixels = torch.sum(first_mask > 0).item()

            if img_area == 0:
                calculated_mask_area_percentage = 0.0
            else:
                calculated_mask_area_percentage = (active_mask_pixels / img_area) * 100.0

        # === Dynamic values based on the *calculated* mask_area_percentage ===
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

        # === Determine target_size based on model_type ===
        if model_type == "Flux":
            target_size_val = 1024
        elif model_type == "SD 1.5":
            target_size_val = 512
        else:
            # Fallback, хотя ComfyUI должен передавать одно из определенных значений
            print(f"Warning: Unknown model_type '{model_type}'. Defaulting target_size to 512.")
            target_size_val = 512

        return (
            image,
            mask,
            context_factor_val,
            expand_px_val,
            blend_px_val,
            target_size_val, # Добавляем новый параметр на выход
        )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AdaptiveParamsWithModelChoice": AdaptiveParamsWithModelChoice
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveParamsWithModelChoice": "Adaptive Params (Model Choice + Passthrough)"
}