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
            "optional": {
                "force_trim_test": ("BOOLEAN",
                                    {"default": False, "label_on": "Force Trim (Test)", "label_off": "Normal"})
            }
        }

    CATEGORY = "Gayrat"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "IMAGE_FULL", "MASK",)
    OUTPUT_NAMES = ("IMAGE", "IMAGE_FULL", "MASK",)
    FUNCTION = "load_image"

    def load_image(self, image, force_trim_test=False):
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

                # Create mask for ComfyUI (inverted)
                mask_np = 1.0 - alpha_np
                mask_tensor = torch.from_numpy(mask_np)[None,]

                # For standard LoadImage compatibility, we need to trim EMPTY space
                # In ComfyUI, LoadImage uses a different approach than simple getbbox
                # It removes rows/columns that are completely transparent

                # Find bounds of non-transparent content
                # Check each row and column for any non-transparent pixel
                rows_with_content = np.any(alpha_np > 0, axis=1)
                cols_with_content = np.any(alpha_np > 0, axis=0)

                if np.any(rows_with_content) and np.any(cols_with_content):
                    # Find first and last row/column with content
                    top = np.argmax(rows_with_content)
                    bottom = len(rows_with_content) - np.argmax(rows_with_content[::-1])
                    left = np.argmax(cols_with_content)
                    right = len(cols_with_content) - np.argmax(cols_with_content[::-1])

                    bbox = (left, top, right, bottom)

                    # For testing: force trim even if edges are not transparent
                    if force_trim_test and bbox == (0, 0, alpha.width, alpha.height):
                        # Trim 10% from each side for testing
                        margin = 0.1
                        left = int(alpha.width * margin)
                        top = int(alpha.height * margin)
                        right = int(alpha.width * (1 - margin))
                        bottom = int(alpha.height * (1 - margin))
                        bbox = (left, top, right, bottom)
                        print(f"[LoadImageWithTrimOptions] FORCE TRIM TEST: trimming to {bbox}")

                    if bbox != (0, 0, alpha.width, alpha.height):
                        # Trim needed
                        print(f"[LoadImageWithTrimOptions] Trimming to bbox: {bbox}")

                        # Crop RGB image
                        image_trimmed = image_rgb.crop(bbox)
                        image_trimmed_np = np.array(image_trimmed).astype(np.float32) / 255.0
                        image_trimmed_tensor = torch.from_numpy(image_trimmed_np)[None,]

                        # Crop mask
                        mask_trimmed = mask_np[top:bottom, left:right]
                        mask_trimmed_tensor = torch.from_numpy(mask_trimmed)[None,]
                    else:
                        # No trim needed
                        print(f"[LoadImageWithTrimOptions] No trimming needed")
                        image_trimmed_tensor = image_full_tensor
                        mask_trimmed_tensor = mask_tensor
                else:
                    # Completely transparent image
                    print(f"[LoadImageWithTrimOptions] Completely transparent image")
                    image_trimmed_tensor = image_full_tensor
                    mask_trimmed_tensor = mask_tensor
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
            # output_image_full = torch.cat(output_images_full, dim=0)
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