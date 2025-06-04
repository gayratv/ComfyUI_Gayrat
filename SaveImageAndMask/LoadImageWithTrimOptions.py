import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import hashlib


class LoadImageWithTrimOptions:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
        }

    CATEGORY = "Gayrat"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "IMAGE_FULL", "MASK",)
    OUTPUT_NAMES = ("IMAGE", "IMAGE_FULL", "MASK",)
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        output_images = []
        output_images_full = []
        output_masks = []

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))

            # Always convert to RGB for full image
            image_rgb = i.convert("RGB")

            # Get full image tensor
            image_full_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_full_tensor = torch.from_numpy(image_full_np)[None,]

            # Handle alpha channel and trimming
            if 'A' in i.getbands():
                # Get alpha channel
                alpha = i.getchannel('A')
                alpha_np = np.array(alpha).astype(np.float32) / 255.0

                # Debug info
                print(f"\n[LoadImageWithTrimOptions] Processing image with alpha channel")
                print(f"[LoadImageWithTrimOptions] Alpha min: {alpha_np.min()}, max: {alpha_np.max()}")

                # Find bounding box of non-transparent area
                alpha_pil = Image.fromarray((alpha_np * 255).astype(np.uint8), mode='L')
                bbox = alpha_pil.getbbox()

                if bbox and bbox != (0, 0, alpha_pil.width, alpha_pil.height):
                    # Trim needed
                    print(f"[LoadImageWithTrimOptions] Trimming to bbox: {bbox}")

                    # Crop RGB image
                    image_trimmed = image_rgb.crop(bbox)
                    image_trimmed_np = np.array(image_trimmed).astype(np.float32) / 255.0
                    image_trimmed_tensor = torch.from_numpy(image_trimmed_np)[None,]

                    # Crop alpha and invert for mask
                    alpha_trimmed = alpha.crop(bbox)
                    alpha_trimmed_np = np.array(alpha_trimmed).astype(np.float32) / 255.0
                    mask_trimmed = 1.0 - alpha_trimmed_np  # Invert for ComfyUI
                    mask_trimmed_tensor = torch.from_numpy(mask_trimmed)[None,]
                else:
                    # No trim needed
                    print(f"[LoadImageWithTrimOptions] No trimming needed")
                    image_trimmed_tensor = image_full_tensor
                    mask_trimmed = 1.0 - alpha_np  # Invert for ComfyUI
                    mask_trimmed_tensor = torch.from_numpy(mask_trimmed)[None,]
            else:
                # No alpha channel
                print(f"\n[LoadImageWithTrimOptions] No alpha channel found")
                image_trimmed_tensor = image_full_tensor
                # Create opaque mask (zeros in ComfyUI format)
                mask_trimmed_tensor = torch.zeros((1, image_full_tensor.shape[1], image_full_tensor.shape[2]),
                                                  dtype=torch.float32)

            # Debug output shapes
            print(f"[LoadImageWithTrimOptions] Full image shape: {image_full_tensor.shape}")
            print(f"[LoadImageWithTrimOptions] Trimmed image shape: {image_trimmed_tensor.shape}")
            print(f"[LoadImageWithTrimOptions] Mask shape: {mask_trimmed_tensor.shape}\n")

            output_images.append(image_trimmed_tensor)
            output_images_full.append(image_full_tensor)
            output_masks.append(mask_trimmed_tensor)

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_image_full = torch.cat(output_images_full, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_image_full = output_images_full[0]
            output_mask = output_masks[0]

        return (output_image, output_image_full, output_mask)

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


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LoadImageWithTrimOptions": LoadImageWithTrimOptions
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithTrimOptions": "Load Image (Trim + Full)"
}