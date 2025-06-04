import torch
import os
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import hashlib

# Импорты, специфичные для ComfyUI
import folder_paths
import node_helpers


class CustomLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Gayrat"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None  # Переменные для хранения эталонной ширины и высоты

        excluded_formats = ['MPO']

        for i_frame in ImageSequence.Iterator(img):
            current_frame_pil = node_helpers.pillow(ImageOps.exif_transpose, i_frame)

            if current_frame_pil.mode == 'I':
                current_frame_pil = current_frame_pil.point(lambda p: p * (1 / 255.0))

            rgb_image_pil = current_frame_pil.convert("RGB")

            if len(output_images) == 0:  # Для первого кадра устанавливаем эталонные размеры
                w = rgb_image_pil.size[0]
                h = rgb_image_pil.size[1]

            # Пропускаем кадры, если их размеры не совпадают с первым
            if rgb_image_pil.size[0] != w or rgb_image_pil.size[1] != h:
                continue

            numpy_image = np.array(rgb_image_pil).astype(np.float32) / 255.0
            torch_image = torch.from_numpy(numpy_image)[None,]

            if 'A' in current_frame_pil.getbands():
                mask_numpy = np.array(current_frame_pil.getchannel('A')).astype(np.float32) / 255.0
                # Инвертируем: 0.0 = полностью видимый, 1.0 = полностью маскированный (прозрачный)
                torch_mask = 1. - torch.from_numpy(mask_numpy)
            elif current_frame_pil.mode == 'P' and 'transparency' in current_frame_pil.info:
                rgba_image_pil_for_mask = current_frame_pil.convert('RGBA')
                mask_numpy = np.array(rgba_image_pil_for_mask.getchannel('A')).astype(np.float32) / 255.0
                torch_mask = 1. - torch.from_numpy(mask_numpy)
            else:  # Если нет альфа-канала, создаем маску, где все видимо
                # Получаем размеры текущего кадра (высота, ширина)
                # rgb_image_pil.size это (width, height)
                # h и w уже были установлены для текущего кадра
                # Маска должна быть (height, width)
                # Создаем маску из нулей (все пиксели видимы/не маскированы)
                torch_mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_images.append(torch_image)
            # Маска должна иметь batch dimension, как и изображение
            output_masks.append(torch_mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            final_output_image = torch.cat(output_images, dim=0)
            final_output_mask = torch.cat(output_masks, dim=0)
        else:
            final_output_image = output_images[0]
            final_output_mask = output_masks[0]

        return (final_output_image, final_output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODE_CLASS_MAPPINGS = {
    "LoadImageCustomFromFileWithSizeCorrectMask": CustomLoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageCustomFromFileWithSizeCorrectMask": "Load Image (Custom, Correct Mask Size)"
}

