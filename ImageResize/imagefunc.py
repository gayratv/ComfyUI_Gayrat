# файл imagefunc.py
import torch
import numpy as np
from PIL import Image

def log(message: str, message_type: str = 'info'):
    name = 'ImageResize'
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# 🐂 ImageResize: {name} -> {message}")

def pil2tensor(image: Image) -> torch.Tensor:
    """Convert a PIL Image to a normalized torch Tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    """Convert a torch Tensor to a PIL Image."""
    array = np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(array)

def image2mask(image: Image) -> torch.Tensor:
    """Convert a PIL mask image to a torch Tensor mask."""
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        # use the red channel for mask
        gray = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(gray)[0, :, :].tolist()])

def num_round_up_to_multiple(number: int, multiple: int) -> int:
    """Round up `number` to the nearest multiple of `multiple`."""
    remainder = number % multiple
    if remainder == 0:
        return number
    return ((number + multiple - 1) // multiple) * multiple

def fit_resize_image(
    image: Image,
    target_width: int,
    target_height: int,
    fit: str,
    resize_sampler,
    background_color: str = '#000000'
) -> Image:
    """
    Resize `image` to (`target_width`, `target_height`) using the given `fit` mode and sampler.
    `fit` can be 'letterbox', 'crop', or 'fill'.
    """
    image = image.convert('RGB')
    orig_width, orig_height = image.size

    if fit == 'letterbox':
        # maintain aspect ratio with padding
        if orig_width / orig_height > target_width / target_height:
            fit_width = target_width
            fit_height = int(target_width / orig_width * orig_height)
        else:
            fit_height = target_height
            fit_width = int(target_height / orig_height * orig_width)
        fit_img = image.resize((fit_width, fit_height), resize_sampler)
        result = Image.new('RGB', (target_width, target_height), color=background_color)
        result.paste(fit_img, ((target_width - fit_width) // 2, (target_height - fit_height) // 2))
    elif fit == 'crop':
        # maintain aspect ratio by cropping
        if orig_width / orig_height > target_width / target_height:
            crop_width = int(orig_height * target_width / target_height)
            left = (orig_width - crop_width) // 2
            fit_img = image.crop((left, 0, left + crop_width, orig_height))
        else:
            crop_height = int(orig_width * target_height / target_width)
            top = (orig_height - crop_height) // 2
            fit_img = image.crop((0, top, orig_width, top + crop_height))
        result = fit_img.resize((target_width, target_height), resize_sampler)
    else:
        # fill: simple resize
        result = image.resize((target_width, target_height), resize_sampler)

    return result

def is_valid_mask(tensor: torch.Tensor) -> bool:
    """Check if the mask tensor contains any non-zero values."""
    return not bool(torch.all(tensor == 0).item())
