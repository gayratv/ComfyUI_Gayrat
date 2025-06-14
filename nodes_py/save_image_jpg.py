import os
import numpy as np
from PIL import Image
import folder_paths


class SaveImageJPG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        # maintain a counter between calls so filenames keep incrementing
        self.counter = 0

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
        results = []
        q = max(1, min(int(quality), 100))

        for i in range(len(image)):
            # ask ComfyUI for a new base path each time so names change
            full_output_folder, filename, _, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, image[i].shape[1], image[i].shape[0]
            )

            img_tensor = image[i]
            img_np = 255.0 * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # debug information
            print(f"[SaveImageJPG] output_folder={full_output_folder}, filename={filename}, current_counter={self.counter}")

            file = f"{filename}_{self.counter:05}.jpg"
            full_path = os.path.join(full_output_folder, file)

            if os.path.exists(full_path):
                existing = [
                    f for f in os.listdir(full_output_folder)
                    if f.startswith(f"{filename}_") and f.endswith(".jpg")
                ]
                nums = []
                for f in existing:
                    num = f[len(filename) + 1:-4]
                    if num.isdigit():
                        nums.append(int(num))
                if nums:
                    self.counter = max(nums) + 1
                else:
                    self.counter += 1
                file = f"{filename}_{self.counter:05}.jpg"
                full_path = os.path.join(full_output_folder, file)

            img_pil.save(full_path, format="JPEG", quality=q)

            print(f"Изображение сохранено: {full_path}")

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

            self.counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "SaveImageJPG": SaveImageJPG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageJPG": "Save Image JPG",
}