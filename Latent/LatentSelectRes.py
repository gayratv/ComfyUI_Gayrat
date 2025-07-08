# gayrat
import torch
import random
import comfy.model_management


# --------------------------------------------------------------------------------
# Gayrat: Вспомогательная функция, встроенная в файл, чтобы убрать внешние зависимости
# --------------------------------------------------------------------------------
def get_size_by_name(name: str) -> dict:
    """
    Возвращает словарь с предустановленными разрешениями для указанной модели.
    """
    sizes = {
        "sd15": {
            "1:1": ["512x512", "640x640", "768x768"],
            "4:3": ["512x384", "768x576", "1024x768"],
            "3:2": ["512x344", "768x512", "960x640"],
            "16:9": ["512x288", "768x432", "1024x576"],
            "21:9": ["768x320", "1024x432"],
            "3:4": ["384x512", "576x768", "768x1024"],
            "2:3": ["344x512", "512x768", "640x960"],
            "9:16": ["288x512", "432x768", "576x1024"],
            "9:21": ["320x768", "432x1024"],
        },
        "sdxl": {
            "1:1": ["1024x1024", "896x896", "768x768", "512x512"],
            "4:3": ["1152x896", "1024x768"],
            "3:2": ["1216x832", "1024x688", "896x592"],
            "16:9": ["1344x768", "1216x688", "1024x576"],
            "21:9": ["1536x640", "1280x544"],
            "3:4": ["896x1152", "768x1024"],
            "2:3": ["832x1216", "688x1024", "592x896"],
            "9:16": ["768x1344", "688x1216", "576x1024"],
            "9:21": ["640x1536", "544x1280"],
        },
        "flux": {
            "1:1": ["1024x1024", "896x896", "768x768"],
            "4:3": ["1152x896", "1024x768"],
            "3:2": ["1216x832", "1024x688"],
            "16:9": ["1344x768", "1024x576"],
            "3:4": ["896x1152", "768x1024"],
            "2:3": ["832x1216", "688x1024"],
            "9:16": ["768x1344", "576x1024"],
        }
    }
    # Для SD3 используем те же размеры, что и для SDXL
    sizes["sd3"] = sizes["sdxl"]

    return sizes.get(name, sizes["sdxl"])  # По умолчанию возвращаем размеры SDXL


# --------------------------------------------------------------------------------
# Основной класс узла
# --------------------------------------------------------------------------------
class FluxSDLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["Flux", "SD3", "SDXL", "SD"],),
            "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "seed_": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "seed_control": (["randomize", "increment", "decrement", "fixed"],),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/gen"

    def generate(self, model, width, height, batch_size=1, seed_=0, seed_control="fixed"):
        if seed_control == "increment":
            seed_ += 1
        elif seed_control == "decrement":
            seed_ -= 1
        elif seed_control == "randomize":
            seed_ = random.randint(0, 0xffffffffffffffff)

        if model == "Flux":
            generator = torch.manual_seed(seed_)
            latent = torch.randn([batch_size, 16, height // 8, width // 8], dtype=torch.float16, device="cpu",
                                 generator=generator).contiguous()
        elif model == "SD3":
            latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        else:  # SD и SDXL
            latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples": latent, "seed_": seed_},)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        if kwargs.get("seed_control") != "fixed":
            return float("NaN")
        return None


# Словарь с размерами для UI, который будет передан в JS
MODEL_SIZES = {
    "SD": get_size_by_name("sd15"),
    "SDXL": get_size_by_name("sdxl"),
    "Flux": get_size_by_name("flux"),
    "SD3": get_size_by_name("sd3")
}