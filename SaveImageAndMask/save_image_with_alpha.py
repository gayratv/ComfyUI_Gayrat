import torch
import numpy as np
from PIL import Image
import os
import folder_paths


class SaveImageWithAlpha:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Gayrat"

    def save_images(self, image, mask, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, image[0].shape[1], image[0].shape[0]
        )
        results = list()

        # Debug output
        print(f"\n[SaveImageWithAlpha] Image shape: {image.shape}")
        print(f"[SaveImageWithAlpha] Mask shape: {mask.shape}")
        print(f"[SaveImageWithAlpha] Number of images: {len(image)}")
        print(f"[SaveImageWithAlpha] Number of masks: {len(mask)}\n")

        # Ensure mask has same batch size as image
        if mask.shape[0] < image.shape[0]:
            mask = mask.repeat(image.shape[0], 1, 1)
        elif mask.shape[0] > image.shape[0]:
            mask = mask[:image.shape[0]]

        for i in range(len(image)):
            img_tensor = image[i]
            mask_tensor = mask[i]

            # Convert image tensor to numpy array (H, W, C)
            img_np = 255. * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # Convert mask to numpy array
            if len(mask_tensor.shape) == 3:
                mask_tensor = mask_tensor.squeeze()
            mask_np = 255. * mask_tensor.cpu().numpy()
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)

            # Create RGBA image
            height, width = img_np.shape[:2]
            rgba_np = np.zeros((height, width, 4), dtype=np.uint8)

            # Copy RGB channels
            rgba_np[:, :, :3] = img_np

            # Add mask as alpha channel
            rgba_np[:, :, 3] = mask_np

            # Convert to PIL Image
            img_pil = Image.fromarray(rgba_np, mode='RGBA')

            # Save the image
            file = f"{filename}_{counter:05}_.png"
            full_path = os.path.join(full_output_folder, file)
            img_pil.save(full_path, compress_level=4)

            # Debug: print saved image info
            print(f"\n[SaveImageWithAlpha] Saved image: {file}")
            print(f"[SaveImageWithAlpha] Saved image size: {img_pil.size}")
            print(f"[SaveImageWithAlpha] Saved image mode: {img_pil.mode}\n")

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SaveImageWithAlpha": SaveImageWithAlpha
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithAlpha": "Save Image with Alpha"
}