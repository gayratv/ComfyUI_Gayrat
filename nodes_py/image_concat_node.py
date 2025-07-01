import torch
from PIL import Image
import numpy as np
import os
import random
import folder_paths

class ConcatImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),

                "background_color": ("STRING", {"default": "#000000"}),
                "layout": (["1*4", "1*3", "2*2"], {"default": "2*2"}),

            },
            "optional": {
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "Gayrat/image processing"

    @staticmethod
    def _tensor_to_pil(t):
        if t.dim() == 4:
            t = t[0]
        array = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        return t.unsqueeze(0)

    @staticmethod
    def _parse_color(color_str: str):
        if isinstance(color_str, str) and color_str.startswith('#') and len(color_str) == 7:
            try:
                return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
            except Exception:
                pass
        return (0, 0, 0)

    def concat(self, image1, image2, background_color, layout, image3=None, image4=None):
        imgs = [image1, image2]
        if image3 is not None:
            imgs.append(image3)
        if image4 is not None:
            imgs.append(image4)

        pil_images = [self._tensor_to_pil(img) for img in imgs]
        bg_color = self._parse_color(background_color)

        max_w = max(img.width for img in pil_images)
        max_h = max(img.height for img in pil_images)

        if layout == "1*4":
            count = min(len(pil_images), 4)
            new_im = Image.new('RGB', (max_w * count, max_h), color=bg_color)
            for idx in range(count):
                img = pil_images[idx]
                mask = img if img.mode in ("RGBA", "LA") else None
                new_im.paste(img, (max_w * idx, 0), mask)
        elif layout == "1*3":
            if image3 is None or image4 is not None:
                raise ValueError("Layout '1*3' requires image3 and no image4")
            new_im = Image.new('RGB', (max_w * 3, max_h), color=bg_color)
            for idx in range(3):
                img = pil_images[idx]
                mask = img if img.mode in ("RGBA", "LA") else None
                new_im.paste(img, (max_w * idx, 0), mask)
        elif layout == "2*2":
            if len(pil_images) == 2:
                new_im = Image.new('RGB', (max_w * 2, max_h), color=bg_color)
                mask0 = pil_images[0] if pil_images[0].mode in ("RGBA", "LA") else None
                mask1 = pil_images[1] if pil_images[1].mode in ("RGBA", "LA") else None
                new_im.paste(pil_images[0], (0, 0), mask0)
                new_im.paste(pil_images[1], (max_w, 0), mask1)
            elif len(pil_images) in (3, 4):
                new_im = Image.new('RGB', (max_w * 2, max_h * 2), color=bg_color)
                mask0 = pil_images[0] if pil_images[0].mode in ("RGBA", "LA") else None
                mask1 = pil_images[1] if pil_images[1].mode in ("RGBA", "LA") else None
                new_im.paste(pil_images[0], (0, 0), mask0)
                new_im.paste(pil_images[1], (max_w, 0), mask1)
                if len(pil_images) >= 3:
                    img2 = pil_images[2]
                    mask2 = img2 if img2.mode in ("RGBA", "LA") else None
                    new_im.paste(img2, (0, max_h), mask2)
                if len(pil_images) == 4:
                    img3 = pil_images[3]
                    mask3 = img3 if img3.mode in ("RGBA", "LA") else None
                    new_im.paste(img3, (max_w, max_h), mask3)
            else:
                raise ValueError("Expected 2 to 4 images")
        else:
            raise ValueError("Unexpected layout value")

        tensor = self._pil_to_tensor(new_im)

        # save preview image similar to PreviewImage node
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"concat_{random.randint(0, 99999999):08d}.png"
        filepath = os.path.join(temp_dir, filename)
        new_im.save(filepath)

        ui = {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}

        return (tensor, {"ui": ui})


NODE_CLASS_MAPPINGS = {
    "ConcatImages": ConcatImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatImages": "Concat Images",
}
