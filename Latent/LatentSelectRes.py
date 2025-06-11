import torch
import numpy as np

"""
Для экономии памяти:
width: 896
height: 896
"""


class FluxSDLatentImageGayrat:
    """
    Custom node for creating empty latent images for Flux, SD and SDXL models
    with proper initialization and aspect ratio support
    """
    # Model-specific size configurations
    MODEL_SIZES = {
        "SD": {
            "1:1": ["512x512", "640x640", "768x768"],
            "4:3": ["512x384", "640x480", "768x576"],
            "3:4": ["384x512", "480x640", "576x768"],
            "16:9": ["512x288", "640x360", "768x432", "896x512"],
            "9:16": ["288x512", "360x640", "432x768", "512x896"]
        },
        "SDXL": {
            "1:1": ["768x768", "896x896", "1024x1024", "1152x1152"],
            "4:3": ["768x576", "896x672", "1024x768", "1152x864"],
            "3:4": ["576x768", "672x896", "768x1024", "864x1152"],
            "16:9": ["768x432", "896x512", "1024x576", "1152x640", "1280x720"],
            "9:16": ["432x768", "512x896", "576x1024", "640x1152", "720x1280"]
        },
        "Flux": {
            # 896x896 - оптимальный размер для экономии памяти
            "1:1": ["896x896", "1024x1024", "1152x1152", "1280x1280", "1408x1408"],
            "4:3": ["896x672", "1024x768", "1152x864", "1280x960", "1408x1056"],
            "3:4": ["672x896", "768x1024", "864x1152", "960x1280", "1056x1408"],
            "16:9": ["896x512", "1024x576", "1152x640", "1280x720", "1408x800", "1536x864", "1920x1088"],
            "9:16": ["512x896", "576x1024", "640x1152", "720x1280", "800x1408", "864x1536", "1088x1920"]
        }
    }
    # Default sizes for each model (used as fallback)
    DEFAULT_SIZES = {
        "SD": "512x512",
        "SDXL": "1024x1024",
        "Flux": "896x896"  # Оптимально для экономии памяти
    }
    # Model configurations
    MODEL_CONFIGS = {
        "SD": {"channels": 4, "init": "constant", "dtype": torch.float32},
        "SDXL": {"channels": 4, "init": "constant", "dtype": torch.float32},
        "Flux": {"channels": 16, "init": "random", "dtype": torch.float16}
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODEL_CONFIGS.keys()),),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16"],),
                "size": (cls.MODEL_SIZES["Flux"]["1:1"],),  # Default to Flux 1:1 sizes
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "seed_control": (["randomize", "fixed", "increment", "decrement"], {
                    "default": "randomize"
                })
            }
        }

    RETURN_TYPES = ("INT", "INT", "LATENT", "INT")
    RETURN_NAMES = ("width", "height", "latent", "seed")
    FUNCTION = "generate_latent"
    CATEGORY = "Gayrat/latent"

    @classmethod
    def IS_CHANGED(cls, model, aspect_ratio, size, batch_size, seed, seed_control):
        # Force update when control is not "fixed"
        if seed_control != "fixed":
            return float("nan")  # Always update
        return f"{model}_{aspect_ratio}_{size}_{batch_size}_{seed}_{seed_control}"

    def generate_latent(self, model, aspect_ratio, size, batch_size, seed, seed_control):
        # Handle seed control logic
        if seed_control == "randomize":
            # Generate new random seed
            import random
            seed = random.randint(0, 0xffffffffffffffff)
        elif seed_control == "increment":
            seed = seed + 1
        elif seed_control == "decrement":
            seed = max(0, seed - 1)
        # "fixed" keeps the original seed value

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
            # For SD/SDXL models - use constant initialization
            latent = torch.ones(
                [batch_size, channels, latent_height, latent_width],
                device=self.device,
                dtype=dtype
            ) * 0.0609

        # Create output dictionary in ComfyUI format
        samples = {"samples": latent}

        return (width, height, samples, seed)

    @classmethod
    def VALIDATE_INPUTS(cls, model, aspect_ratio, size, batch_size, seed, seed_control):
        # Validate that the selected size is appropriate for the model
        valid_sizes = cls.MODEL_SIZES.get(model, {}).get(aspect_ratio, [])

        if size not in valid_sizes:
            # Check if size exists in any model's configuration for this aspect ratio
            for m, sizes in cls.MODEL_SIZES.items():
                if size in sizes.get(aspect_ratio, []):
                    return f"Size {size} is not recommended for {model} model. Consider using sizes: {', '.join(valid_sizes)}"
            return f"Size {size} is not valid for aspect ratio {aspect_ratio}"

        return True


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "FluxSDLatentImageGayrat": FluxSDLatentImageGayrat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxSDLatentImageGayrat": "Flux/SD Latent Image Gayrat"
}