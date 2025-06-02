import torch

# Constants for interpolation and preresize modes, matching CropAndStitch where possible
INTERPOLATION_MODES = ["bicubic", "bilinear", "area", "nearest", "lanczos"]
PRERESIZE_MODES = ["ensure minimum resolution", "ensure maximum resolution", "just resize"]
MAX_RESOLUTION_DEFAULT = 16384  # A common default for max resolution in ComfyUI nodes
PADDING_VALUES = ["0", "8", "16", "32", "64", "128", "256", "512"]

class AdaptiveCropStitchParamsV2:
    """
    A node to dynamically generate parameters for InpaintCropAndStitch
    by first calculating the mask area percentage relative to the image,
    and then applying predefined recommendations.
    """

    @classmethod
    def INPUT_TYPES(s):
        # Define the list of allowed padding values for the dropdown

        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                # Allow user to override key "fixed" parameters based on my recommendations
                "target_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION_DEFAULT, "step": 32}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION_DEFAULT, "step": 32}),
                # Modified target_padding to use a dropdown list
                # This ensures that target_padding (and thus output_padding) is one of these fixed values.
                "target_padding": (PADDING_VALUES, {"default": "32"}),
                # "upscale_algo": (INTERPOLATION_MODES, {"default": "bicubic"}),
                # "downscale_algo": (INTERPOLATION_MODES, {"default": "bilinear"}),
                "hipass_filter_strength": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "FLOAT",  # calculated_mask_area_percentage
        "FLOAT",  # context_from_mask_extend_factor
        "INT",  # mask_expand_pixels
        "INT",  # mask_blend_pixels
        "INT",  # output_target_width
        "INT",  # output_target_height
        PADDING_VALUES,  # output_padding  <- This will be an INT, one of the values selected from the input dropdown
        "BOOLEAN",  # output_resize_to_target_size
        # "STRING",  # upscale_algorithm
        # "STRING",  # downscale_algorithm
        "BOOLEAN",  # mask_fill_holes
        "BOOLEAN",  # mask_invert
        "FLOAT",  # mask_hipass_filter
        "BOOLEAN",  # preresize
        "STRING",  # preresize_mode
        "INT",  # preresize_min_width
        "INT",  # preresize_min_height
        "INT",  # preresize_max_width
        "INT",  # preresize_max_height
        "BOOLEAN",  # extend_for_outpainting
        "FLOAT",  # extend_up_factor
        "FLOAT",  # extend_down_factor
        "FLOAT",  # extend_left_factor
        "FLOAT",  # extend_right_factor,
    )
    RETURN_NAMES = (
        "calculated_mask_area_percentage",
        "context_from_mask_extend_factor",
        "mask_expand_pixels",
        "mask_blend_pixels",
        "output_target_width",
        "output_target_height",
        "output_padding2",
        "output_resize_to_target_size",
        "upscale_algorithm",
        "downscale_algorithm",
        "mask_fill_holes",
        "mask_invert",
        "mask_hipass_filter",
        "preresize",
        "preresize_mode",
        "preresize_min_width",
        "preresize_min_height",
        "preresize_max_width",
        "preresize_max_height",
        "extend_for_outpainting",
        "extend_up_factor",
        "extend_down_factor",
        "extend_left_factor",
        "extend_right_factor",
    )

    FUNCTION = "get_params"
    CATEGORY = "Gayrat"  # Custom category for organization

    def get_params(self, image: torch.Tensor, mask: torch.Tensor,
                   target_width, target_height, target_padding, # target_padding will be an int from the list
                   upscale_algo, downscale_algo, hipass_filter_strength):

        # === Calculate mask area percentage ===
        # Image shape is typically (Batch, Height, Width, Channels) for ComfyUI IMAGE
        # Mask shape is typically (Batch, Height, Width) for ComfyUI MASK

        if image is None or mask is None:
            # Should not happen if inputs are required, but good practice
            calculated_mask_area_percentage = 0.0
        else:
            # Get dimensions of the first image in the batch
            # ComfyUI IMAGE tensor typically NCHW after conversion from comfy space, so H=shape[2], W=shape[3]
            # Or if it's NHWC (common internal ComfyUI representation), H=shape[1], W=shape[2]
            # Let's assume NHWC for direct image input as per Comfy standard for 'IMAGE' type
            if image.dim() == 4 and image.shape[3] in [1, 3, 4]:  # NHWC
                img_height, img_width = image.shape[1], image.shape[2]
            elif image.dim() == 3:  # Assuming it might be a single NCHW image passed weirdly, or just CHW
                img_height, img_width = image.shape[1], image.shape[2]  # Take H,W
            else:  # Fallback or unexpected format, take first spatial dims
                print(f"Warning: Unexpected image dimensions {image.shape}, attempting to get H, W.")
                img_height, img_width = image.shape[-2], image.shape[-1]

            img_area = img_height * img_width

            # Get the first mask in the batch
            # ComfyUI MASK tensor is typically (Batch, Height, Width)
            # We need to sum non-zero pixels for the first mask in the batch
            # Mask values are typically 0.0 to 1.0
            first_mask = mask[0]  # Shape: (Height, Width)

            # Sum of mask pixels (where mask > 0, assuming mask values are 0 or 1, or thresholded)
            # To be safe, let's consider any positive value as part of the mask
            active_mask_pixels = torch.sum(first_mask > 0).item()

            if img_area == 0:
                calculated_mask_area_percentage = 0.0
            else:
                calculated_mask_area_percentage = (active_mask_pixels / img_area) * 100.0

        # --- The rest of the logic is similar to the previous version ---

        # Fixed values (some are now inputs with defaults, others are my core recommendations)
        output_resize_to_target_size_val = True
        output_target_w_val = target_width
        output_target_h_val = target_height
        # output_pad_val will be the integer value selected from the 'target_padding' dropdown.
        # This ensures it's one of the fixed values [0, 8, 16, 32, 64, 128, 256, 512].
        output_pad_val = target_padding
        upscale_algorithm_val = upscale_algo
        downscale_algorithm_val = downscale_algo
        mask_fill_holes_val = True
        mask_invert_val = False
        mask_hipass_filter_val = hipass_filter_strength

        preresize_val = False
        extend_for_outpainting_val = False

        preresize_mode_val = PRERESIZE_MODES[0]
        preresize_min_w_val = 1024
        preresize_min_h_val = 1024
        preresize_max_w_val = MAX_RESOLUTION_DEFAULT
        preresize_max_h_val = MAX_RESOLUTION_DEFAULT

        extend_up_val = 1.0
        extend_down_val = 1.0
        extend_left_val = 1.0
        extend_right_val = 1.0

        # === Dynamic values based on the *calculated* mask_area_percentage ===
        if calculated_mask_area_percentage < 10.0:  # Small mask
            context_factor_val = 2.0
            expand_px_val = 6
            blend_px_val = 12
        elif calculated_mask_area_percentage < 40.0:  # Medium mask
            context_factor_val = 1.5
            expand_px_val = 12
            blend_px_val = 24
        else:  # Large mask
            context_factor_val = 1.1
            expand_px_val = 24
            blend_px_val = 48

        return (
            calculated_mask_area_percentage,  # New output
            context_factor_val,
            expand_px_val,
            blend_px_val,
            output_target_w_val,
            output_target_h_val,
            output_pad_val, # This is an INT from the fixed list
            output_resize_to_target_size_val,
            # upscale_algorithm_val,
            # downscale_algorithm_val,
            mask_fill_holes_val,
            mask_invert_val,
            mask_hipass_filter_val,
            preresize_val,
            preresize_mode_val,
            preresize_min_w_val,
            preresize_min_h_val,
            preresize_max_w_val,
            preresize_max_h_val,
            extend_for_outpainting_val,
            extend_up_val,
            extend_down_val,
            extend_left_val,
            extend_right_val,
        )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AdaptiveCropStitchParamsV2": AdaptiveCropStitchParamsV2
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveCropStitchParamsV2": "Adaptive Crop & Stitch Params (Calc %)"
}