import torch


class ModelResolutionSelector:
    """
    A custom node that automatically sets resolution based on model type selection
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["Flux", "SD"], {"default": "Flux"}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("resolution", "model_type")
    FUNCTION = "get_resolution"
    CATEGORY = "utils/model"

    def get_resolution(self, model_type):
        # Set resolution based on model type
        if model_type == "Flux":
            resolution = 1024
        else:  # SD
            resolution = 512

        return (resolution, model_type)


class ModelResolutionSelectorAdvanced:
    """
    Advanced version with width and height outputs
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["Flux", "SD"], {"default": "Flux"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "model_type")
    FUNCTION = "get_resolution"
    CATEGORY = "utils/model"

    def get_resolution(self, model_type, custom_width=0, custom_height=0):
        # Use custom dimensions if provided
        if custom_width > 0 and custom_height > 0:
            return (custom_width, custom_height, model_type)

        # Otherwise use default based on model type
        if model_type == "Flux":
            width = height = 1024
        else:  # SD
            width = height = 512

        return (width, height, model_type)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ModelResolutionSelector": ModelResolutionSelector,
    "ModelResolutionSelectorAdvanced": ModelResolutionSelectorAdvanced,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelResolutionSelector": "Model Resolution Selector",
    "ModelResolutionSelectorAdvanced": "Model Resolution Selector (Advanced)",
}