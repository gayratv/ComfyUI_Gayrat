import os
import numpy as np
from PIL import Image
import folder_paths


class SaveImageJPG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Gayrat"

    def save_images(self, image, quality=90, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, image[0].shape[1], image[0].shape[0]
        )

        results = []
        q = max(1, min(int(quality), 100))

        for i in range(len(image)):
            img_tensor = image[i]
            img_np = 255.0 * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            file = f"{filename}_{counter:05}.jpg"
            full_path = os.path.join(full_output_folder, file)
            img_pil.save(full_path, format="JPEG", quality=q)

            print(f"Изображение сохранено: {full_path}")

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "SaveImageJPG": SaveImageJPG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageJPG": "Save Image JPG",
}