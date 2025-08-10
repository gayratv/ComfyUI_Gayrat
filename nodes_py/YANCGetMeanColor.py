"""
fork from
https://github.com/ALatentPlace/ComfyUI_yanc

Полностью нода у меня не запустилась, вырезал один узел
"""

import torch
import folder_paths
from comfy.cli_args import args
from comfy_extras import nodes_mask as masks
import nodes as nodes


class YANCGetMeanColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
                {
                    "image": ("IMAGE",),
                    "amplify": ("BOOLEAN", {"default": False})
                },
            "optional":
                {
                    "mask_opt": ("MASK",),
                },
                }

    CATEGORY = 'Gayrat/images'

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("int", "red", "green", "blue", "hex")
    FUNCTION = "do_it"

    def do_it(self, image, amplify, mask_opt=None):
        masked_image = image.clone()

        if mask_opt is not None:
            if mask_opt.shape[1:3] != image.shape[1:3]:
                raise ValueError(
                    "Mask and image spatial dimensions must match.")

            mask_opt = mask_opt.unsqueeze(-1)
            masked_image = masked_image * mask_opt

            num_masked_pixels = torch.sum(mask_opt)
            if num_masked_pixels == 0:
                raise ValueError(
                    "No masked pixels found in the image. Please set a mask.")

            sum_r = torch.sum(masked_image[:, :, :, 0])
            sum_g = torch.sum(masked_image[:, :, :, 1])
            sum_b = torch.sum(masked_image[:, :, :, 2])

            r_mean = sum_r / num_masked_pixels
            g_mean = sum_g / num_masked_pixels
            b_mean = sum_b / num_masked_pixels
        else:
            r_mean = torch.mean(masked_image[:, :, :, 0])
            g_mean = torch.mean(masked_image[:, :, :, 1])
            b_mean = torch.mean(masked_image[:, :, :, 2])

        r_mean_255 = r_mean.item() * 255.0
        g_mean_255 = g_mean.item() * 255.0
        b_mean_255 = b_mean.item() * 255.0

        if amplify:
            highest_value = max(r_mean_255, g_mean_255, b_mean_255)
            diff_to_max = 255.0 - highest_value

            amp_factor = 1.0

            r_mean_255 += diff_to_max * amp_factor * \
                (r_mean_255 / highest_value)
            g_mean_255 += diff_to_max * amp_factor * \
                (g_mean_255 / highest_value)
            b_mean_255 += diff_to_max * amp_factor * \
                (b_mean_255 / highest_value)

            r_mean_255 = min(max(r_mean_255, 0), 255)
            g_mean_255 = min(max(g_mean_255, 0), 255)
            b_mean_255 = min(max(b_mean_255, 0), 255)

        fill_value = (int(r_mean_255) << 16) + \
            (int(g_mean_255) << 8) + int(b_mean_255)

        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r_mean_255), int(g_mean_255), int(b_mean_255)).upper()

        return (fill_value, int(r_mean_255), int(g_mean_255), int(b_mean_255), hex_color,)



NODE_CLASS_MAPPINGS = {
    "Get_Mean_Color2": YANCGetMeanColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Get_Mean_Color2": "Get Mean Color (Gayrat)"
}