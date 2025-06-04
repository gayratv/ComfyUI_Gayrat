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

            # Get full image
            image_full = i.convert("RGB")
            image_full_np = np.array(image_full).astype(np.float32) / 255.0
            image_full_tensor = torch.from_numpy(image_full_np)[None,]

            # Handle alpha channel and trimming
            if 'A' in i.getbands():
                mask = i.getchannel('A')
                mask_np = np.array(mask).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,]

                # Debug mask values
                print(f"\n[LoadImageWithTrimOptions] Mask min value: {mask_np.min()}")
                print(f"[LoadImageWithTrimOptions] Mask max value: {mask_np.max()}")
                print(f"[LoadImageWithTrimOptions] Mask has transparency: {(mask_np < 1.0).any()}")

                # Trim image by mask (same as standard LoadImage)
                bbox = mask.getbbox()
                if bbox:
                    # Crop image and mask to bounding box
                    image_trimmed = image_full.crop(bbox)
                    mask_trimmed = mask.crop(bbox)

                    image_trimmed_np = np.array(image_trimmed).astype(np.float32) / 255.0
                    image_trimmed_tensor = torch.from_numpy(image_trimmed_np)[None,]

                    mask_trimmed_np = np.array(mask_trimmed).astype(np.float32) / 255.0
                    mask_trimmed_tensor = torch.from_numpy(mask_trimmed_np)[None,]

                    print(f"\n[LoadImageWithTrimOptions] Bounding box found: {bbox}")
                else:
                    # Empty mask, use full image
                    print(f"\n[LoadImageWithTrimOptions] No bounding box found, using full image")
                    image_trimmed_tensor = image_full_tensor
                    mask_trimmed_tensor = mask_tensor
            else:
                # No alpha channel - trimmed and full are the same
                image_trimmed_tensor = image_full_tensor
                mask_tensor = torch.ones((1, image_full_tensor.shape[1], image_full_tensor.shape[2]),
                                         dtype=torch.float32)
                mask_trimmed_tensor = mask_tensor

            output_images.append(image_trimmed_tensor)
            output_images_full.append(image_full_tensor)
            output_masks.append(mask_trimmed_tensor)

            print(f"\n[LoadImageWithTrimOptions] Full image shape: {image_full_tensor.shape}")
            print(f"[LoadImageWithTrimOptions] Trimmed image shape: {image_trimmed_tensor.shape}")
            print(f"[LoadImageWithTrimOptions] Mask shape: {mask_trimmed_tensor.shape}\n")

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