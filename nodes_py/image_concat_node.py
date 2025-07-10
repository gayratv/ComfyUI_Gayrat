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
                "background_color": ("COLOR", {"default": "#000000"}),
                "layout": (["1*1", "1*4", "1*3", "2*2"], {"default": "2*2"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "Gayrat/image processing"

    # ---------- helpers ----------
    @staticmethod
    def _tensor_to_pil(t):
        if t.dim() == 4:
            t = t[0]
        arr = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        return t.unsqueeze(0)

    @staticmethod
    def _parse_color(c: str):
        if isinstance(c, str) and c.startswith("#") and len(c) == 7:
            try:
                return tuple(int(c[i : i + 2], 16) for i in (1, 3, 5))
            except Exception:
                pass
        return (0, 0, 0)

    # ---------- main ----------
    def concat(
        self,
        image1,
        background_color,
        layout,
        image2=None,
        image3=None,
        image4=None,
    ):
        imgs = [img for img in (image1, image2, image3, image4) if img is not None]

        # sanity-checks для разных схем
        if layout == "1*1" and len(imgs) < 1:
            raise ValueError("Layout '1*1' требует хотя бы image1")
        if layout == "1*4" and len(imgs) < 2:
            raise ValueError("Layout '1*4' требует ≥ 2 изображений")
        if layout == "1*3" and len(imgs) != 3:
            raise ValueError("Layout '1*3' требует ровно 3 изображения")
        if layout == "2*2" and not (2 <= len(imgs) <= 4):
            raise ValueError("Layout '2*2' требует 2–4 изображений")

        pil_images = [self._tensor_to_pil(i) for i in imgs]
        bg_color = self._parse_color(background_color)

        max_w = max(p.width for p in pil_images)
        max_h = max(p.height for p in pil_images)

        # ---------- компоновка ----------
        if layout == "1*1":
            new_im = pil_images[0].copy()

        elif layout == "1*4":
            count = min(len(pil_images), 4)
            new_im = Image.new("RGB", (max_w * count, max_h), color=bg_color)
            for idx in range(count):
                img, mask = pil_images[idx], None
                if img.mode in ("RGBA", "LA"):
                    mask = img
                new_im.paste(img, (max_w * idx, 0), mask)

        elif layout == "1*3":
            new_im = Image.new("RGB", (max_w * 3, max_h), color=bg_color)
            for idx in range(3):
                img, mask = pil_images[idx], None
                if img.mode in ("RGBA", "LA"):
                    mask = img
                new_im.paste(img, (max_w * idx, 0), mask)

        elif layout == "2*2":
            if len(pil_images) == 2:
                new_im = Image.new("RGB", (max_w * 2, max_h), color=bg_color)
                for idx in range(2):
                    img, mask = pil_images[idx], None
                    if img.mode in ("RGBA", "LA"):
                        mask = img
                    new_im.paste(img, (max_w * idx, 0), mask)
            else:
                new_im = Image.new("RGB", (max_w * 2, max_h * 2), color=bg_color)
                positions = [(0, 0), (max_w, 0), (0, max_h), (max_w, max_h)]
                for idx, pos in enumerate(positions[: len(pil_images)]):
                    img, mask = pil_images[idx], None
                    if img.mode in ("RGBA", "LA"):
                        mask = img
                    new_im.paste(img, pos, mask)
        else:
            raise ValueError("Неизвестное значение layout")

        # ---------- вывод ----------
        tensor = self._pil_to_tensor(new_im)

        # превью (как PreviewImage)
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"concat_{random.randint(0, 99999999):08d}.png"
        new_im.save(os.path.join(temp_dir, filename))

        ui = {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}

        return (tensor, {"ui": ui})


NODE_CLASS_MAPPINGS = {"ConcatImages": ConcatImages}
NODE_DISPLAY_NAME_MAPPINGS = {"ConcatImages": "Concat Images"}
