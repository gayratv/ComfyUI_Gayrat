import torch
import numpy as np

"""
Для экономии памяти:
width: 896
height: 896
"""

class FluxSDLatentImage:
    """
    Custom node for creating empty latent images for Flux and SD models
    with proper initialization and aspect ratio support
    """

    # Define available sizes for each aspect ratio (all divisible by 32)
    SIZES = {
        "1:1": [
            "512x512", "640x640", "768x768", "896x896",
            "1024x1024", "1152x1152", "1280x1280", "1408x1408"
        ],
        "4:3": [
            "512x384", "640x480", "768x576", "896x672",
            "1024x768", "1152x864", "1280x960", "1408x1056"
        ],
        "3:4": [
            "384x512", "480x640", "576x768", "672x896",
            "768x1024", "864x1152", "960x1280", "1056x1408"
        ],
        "16:9": [
            "512x288", "640x360", "768x432", "896x512",
            "1024x576", "1152x640", "1280x720", "1408x800",
            "1536x864", "1664x960", "1920x1088"
        ],
        "9:16": [
            "288x512", "360x640", "432x768", "512x896",
            "576x1024", "640x1152", "720x1280", "800x1408",
            "864x1536", "960x1664", "1088x1920"
        ]
    }

    # Model configurations
    MODEL_CONFIGS = {
        "Flux": {"channels": 16, "init": "random", "dtype": torch.float16},
        "Flux PRO": {"channels": 16, "init": "random", "dtype": torch.float16},
        "Flux Ultra": {"channels": 16, "init": "random", "dtype": torch.float16},
        "SD": {"channels": 4, "init": "constant", "dtype": torch.float32}
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODEL_CONFIGS.keys()),),
                "aspect_ratio": (list(cls.SIZES.keys()),),
                "size": (cls.SIZES["1:1"],),  # Default to 1:1 sizes
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                })
            }
        }

    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "latent")
    FUNCTION = "generate_latent"
    CATEGORY = "latent"

    @classmethod
    def IS_CHANGED(cls, model, aspect_ratio, size, batch_size, seed):
        # This ensures the node updates when inputs change
        return f"{model}_{aspect_ratio}_{size}_{batch_size}_{seed}"

    def generate_latent(self, model, aspect_ratio, size, batch_size, seed):
        # Parse size string to get width and height
        width, height = map(int, size.split('x'))

        # Get model configuration
        config = self.MODEL_CONFIGS[model]
        channels = config["channels"]
        init_type = config["init"]
        dtype = config["dtype"]

        # Calculate latent dimensions (divide by 8 for VAE encoding)
        latent_height = height // 8
        latent_width = width // 8

        # Create latent tensor based on model type
        if init_type == "random":
            # For Flux models - use random noise
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latent = torch.randn(
                [batch_size, channels, latent_height, latent_width],
                generator=generator,
                device=self.device,
                dtype=dtype
            )
        else:
            # For SD models - use constant initialization
            latent = torch.ones(
                [batch_size, channels, latent_height, latent_width],
                device=self.device,
                dtype=dtype
            ) * 0.0609

        # Create output dictionary in ComfyUI format
        samples = {"samples": latent}

        return (width, height, samples)

    @classmethod
    def VALIDATE_INPUTS(cls, model, aspect_ratio, size, batch_size, seed):
        # Validate that the selected size matches the aspect ratio
        if size not in cls.SIZES.get(aspect_ratio, []):
            return f"Size {size} is not valid for aspect ratio {aspect_ratio}"
        return True


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "FluxSDLatentImage": FluxSDLatentImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxSDLatentImage": "Flux/SD Latent Image"
}


# Additional helper class for dynamic size updates (optional)
class FluxSDLatentImageAdvanced(FluxSDLatentImage):
    """
    Advanced version with dynamic size list updates based on aspect ratio
    This requires frontend JavaScript modifications for full functionality
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODEL_CONFIGS.keys()),),
                "aspect_ratio": (list(cls.SIZES.keys()), {
                    "default": "1:1"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                })
            },
            "optional": {
                "size_override": ("STRING", {
                    "default": "",
                    "multiline": False
                })
            }
        }

    def generate_latent(self, model, aspect_ratio, batch_size, seed, size_override=""):
        # Use size_override if provided, otherwise use default for aspect ratio
        if size_override and 'x' in size_override:
            size = size_override
        else:
            # Get default size for the aspect ratio
            sizes = self.SIZES.get(aspect_ratio, ["1024x1024"])
            # Choose appropriate default based on model
            if model.startswith("Flux"):
                # For Flux, prefer 1024x1024 or closest
                size = next((s for s in sizes if "1024" in s), sizes[len(sizes) // 2])
            else:
                # For SD, prefer 512x512 or closest
                size = next((s for s in sizes if "512" in s), sizes[0])

        return super().generate_latent(model, aspect_ratio, size, batch_size, seed)


# Register the advanced node as well
NODE_CLASS_MAPPINGS["FluxSDLatentImageAdvanced"] = FluxSDLatentImageAdvanced
NODE_DISPLAY_NAME_MAPPINGS["FluxSDLatentImageAdvanced"] = "Flux/SD Latent Image (Advanced)"