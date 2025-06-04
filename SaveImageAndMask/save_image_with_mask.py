import torch
import numpy as np
from PIL import Image
import os
import folder_paths


class SaveImageWithMask:
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

        for (batch_number, img) in enumerate(image):
            # Convert image tensor to numpy array
            i = 255. * img.cpu().numpy()
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Handle mask
            if batch_number < mask.shape[0]:
                mask_slice = mask[batch_number]
            else:
                # If not enough masks, use the last one
                mask_slice = mask[-1]

            # Convert mask to numpy array and ensure it's 2D
            if len(mask_slice.shape) == 3:
                mask_slice = mask_slice.squeeze()

            # Convert mask to 0-255 range
            mask_np = 255. * mask_slice.cpu().numpy()
            mask_img = Image.fromarray(np.clip(mask_np, 0, 255).astype(np.uint8), mode='L')

            # Convert RGB to RGBA and add mask as alpha channel
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')

            # Replace alpha channel with mask
            img_pil.putalpha(mask_img)

            # Save the image
            file = f"{filename}_{counter:05}_.png"
            img_pil.save(os.path.join(full_output_folder, file), compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SaveImageWithMask": SaveImageWithMask
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithMask": "Save Image with Mask"
}