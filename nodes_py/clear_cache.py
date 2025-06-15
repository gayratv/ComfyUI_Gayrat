import gc
import comfy.model_management as mm

def clean_gpu_and_cache():
    """Функция очистки GPU и кэша."""
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()
    print("Cache and GPU cleared.")

class EasyClearGpuAndCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": ("*", {})}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "clear_gpu_and_cache"
    CATEGORY = "Gayrat/cache"

    def clear_gpu_and_cache(self, anything):
        clean_gpu_and_cache()
        return (anything,)


NODE_CLASS_MAPPINGS = {
    "EasyClearGpuAndCache": EasyClearGpuAndCache
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyClearGpuAndCache": "Easy Clear GPU and Cache"
}