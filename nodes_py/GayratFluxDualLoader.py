import torch
import comfy.sd
import folder_paths


# --------------------------------------------------------------------------------
# КЛАСС 1: Улучшенный загрузчик для FLUX
# --------------------------------------------------------------------------------
class GayratFluxDualLoader:
    """
    Улучшенный загрузчик специально для FLUX.
    Принимает на вход имена файлов для CLIP-L и T5 и выдает две
    отдельные, готовые к использованию модели.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_l_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Выберите файл модели CLIP-L для FLUX"
                }),
                "t5_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Выберите файл модели T5 для FLUX"
                }),
            }
        }

    RETURN_TYPES = ("CLIP", "CLIP")
    RETURN_NAMES = ("clip_l", "t5")
    FUNCTION = "load_clips"
    CATEGORY = "Gayrat/conditioning"

    def load_clips(self, clip_l_name, t5_name):
        # Загружаем модель CLIP-L
        clip_l_path = folder_paths.get_full_path("text_encoders", clip_l_name)
        clip_l = comfy.sd.load_clip(
            ckpt_paths=[clip_l_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.FLUX
        )

        # Загружаем модель T5
        t5_path = folder_paths.get_full_path("text_encoders", t5_name)
        t5 = comfy.sd.load_clip(
            ckpt_paths=[t5_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.FLUX
        )

        return (clip_l, t5)


# --------------------------------------------------------------------------------
# КЛАСС 2: Кодировщик с раздельными промптами для FLUX
# --------------------------------------------------------------------------------
class FluxSeparatePromptEncoder:
    """
    Продвинутая нода для FLUX, позволяющая подавать отдельные промпты
    на отдельно загруженные кодировщики CLIP-L и T5.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_l": ("CLIP", {"tooltip": "Модель CLIP-L, загруженная через GayratFluxDualLoader"}),
                "t5": ("CLIP", {"tooltip": "Модель T5, загруженная через GayratFluxDualLoader"}),
                "prompt_for_clip_l": ("STRING", {
                    "multiline": True,
                    "default": "краткий промпт, стиль, композиция (до 77 токенов)",
                    "tooltip": "Промпт для CLIP-L. Отвечает за стиль и общую композицию. Должен быть коротким."
                }),
                "prompt_for_t5": ("STRING", {
                    "multiline": True,
                    "default": "длинный детальный промпт...",
                    "tooltip": "Промпт для T5. Отвечает за семантику и детали. Может быть очень длинным."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Gayrat/conditioning"

    def encode(self, clip_l, t5, prompt_for_clip_l, prompt_for_t5):
        tokens_l = clip_l.tokenize(prompt_for_clip_l)
        cond_l, pooled_l = clip_l.encode_from_tokens(tokens_l, return_pooled=True)
        tokens_t5 = t5.tokenize(prompt_for_t5)
        cond_t5, pooled_t5 = t5.encode_from_tokens(tokens_t5, return_pooled=True)
        final_pooled = torch.cat([pooled_l, pooled_t5], dim=-1)
        final_conditioning = [[cond_l, {"pooled_output": final_pooled}], [cond_t5, {"pooled_output": final_pooled}]]
        return (final_conditioning,)


# --------------------------------------------------------------------------------
# КЛАСС 3: Загрузчик для SDXL (сразу две модели)
# --------------------------------------------------------------------------------
class SDXLDualClipLoader:
    """
    Упрощенный загрузчик для SDXL. Загружает обе модели CLIP (L и G)
    из одного комбинированного файла или двух отдельных.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_g_name": (folder_paths.get_filename_list("text_encoders"),),
                "clip_l_name": (folder_paths.get_filename_list("text_encoders"),),
            }
        }

    RETURN_TYPES = ("CLIP", "CLIP")
    RETURN_NAMES = ("clip_g", "clip_l")
    FUNCTION = "load_sdxl_clip"
    CATEGORY = "Gayrat/conditioning"

    def load_sdxl_clip(self, clip_g_name, clip_l_name):
        clip_g_path = folder_paths.get_full_path("text_encoders", clip_g_name)
        clip_l_path = folder_paths.get_full_path("text_encoders", clip_l_name)

        # Загружаем обе модели с указанием типа SDXL
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_g_path, clip_l_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.SDXL
        )
        return (clip.patcher.model.clip_g, clip.patcher.model.clip_l)


# --------------------------------------------------------------------------------
# КЛАСС 4: Кодировщик с раздельными промптами для SDXL
# --------------------------------------------------------------------------------
class SDXLSeparatePromptEncoder:
    """
    Продвинутая нода для SDXL, позволяющая подавать отдельные промпты
    на кодировщики CLIP-G и CLIP-L, а также задавать параметры разрешения.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_g": ("CLIP",),
                "clip_l": ("CLIP",),
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
                "prompt_for_clip_g": ("STRING",
                                      {"multiline": True, "default": "Промпт для CLIP-G (основные детали)..."}),
                "prompt_for_clip_l": ("STRING",
                                      {"multiline": True, "default": "Промпт для CLIP-L (стиль, вторичные детали)..."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Gayrat/conditioning"

    def encode(self, clip_g, clip_l, width, height, prompt_for_clip_g, prompt_for_clip_l):
        # Кодирование для CLIP-G
        tokens_g = clip_g.tokenize(prompt_for_clip_g)
        cond_g, pooled_g = clip_g.encode_from_tokens(tokens_g, return_pooled=True)

        # Кодирование для CLIP-L
        tokens_l = clip_l.tokenize(prompt_for_clip_l)
        cond_l, _ = clip_l.encode_from_tokens(tokens_l, return_pooled=False)  # pooled_l не используется в SDXL

        # Добавление информации о разрешении к pooled_g
        sdxl_pooled = comfy.sd.encode_sdxl_pool_cond(pooled_g, width, height, 0, 0, width, height)

        # Комбинирование эмбеддингов
        final_cond = torch.cat((cond_g, cond_l), dim=1)

        final_conditioning = [[final_cond, {"pooled_output": sdxl_pooled}]]

        return (final_conditioning,)


# --- Регистрация нод в ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "FluxSeparatePromptEncoder": FluxSeparatePromptEncoder,
    "GayratFluxDualLoader": GayratFluxDualLoader,
    "SDXLDualClipLoader": SDXLDualClipLoader,
    "SDXLSeparatePromptEncoder": SDXLSeparatePromptEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxSeparatePromptEncoder": "Flux Separate Prompt Encoder",
    "GayratFluxDualLoader": "Flux Dual CLIP Loader (Gayrat)",
    "SDXLDualClipLoader": "SDXL Dual CLIP Loader (Gayrat)",
    "SDXLSeparatePromptEncoder": "SDXL Separate Prompt Encoder",
}
