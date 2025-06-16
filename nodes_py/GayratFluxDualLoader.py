import torch
import comfy.sd
import folder_paths
from comfy.text_encoders.flux import FluxClipModel


# --------------------------------------------------------------------------------
# КЛАСС 1: Правильный загрузчик для FLUX (загружает обе модели)
# --------------------------------------------------------------------------------
class GayratFluxDualLoader:
    """
    Правильный загрузчик для FLUX.
    Принимает на вход имена файлов для CLIP-L и T5 и выдает один
    комбинированный объект CLIP, готовый к работе.
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

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clips"
    CATEGORY = "Gayrat/conditioning"

    def load_clips(self, clip_l_name, t5_name):
        clip_l_path = folder_paths.get_full_path("text_encoders", clip_l_name)
        t5_path = folder_paths.get_full_path("text_encoders", t5_name)

        flux_clip = comfy.sd.load_clip(
            ckpt_paths=[clip_l_path, t5_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.FLUX
        )
        return (flux_clip,)


# --------------------------------------------------------------------------------
# КЛАСС 2: Кодировщик с раздельными промптами для FLUX
# --------------------------------------------------------------------------------
class GayratFluxEncoder:
    """
    Продвинутая нода для FLUX, позволяющая подавать отдельные промпты
    на внутренние кодировщики CLIP-L и T5.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Комбинированный CLIP для FLUX, загруженный через GayratFluxDualLoader"}),
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

    def encode(self, clip, prompt_for_clip_l, prompt_for_t5):
        # Проверяем, что на вход подали правильный объект.
        if not hasattr(clip.cond_stage_model, 'clip_l') or not hasattr(clip.cond_stage_model, 't5xxl'):
            raise TypeError(
                "Эта нода предназначена для работы только с моделью FLUX. Убедитесь, что вы используете 'GayratFluxDualLoader'.")

        # "Вытаскиваем" внутренние МОДЕЛИ
        clip_l_model = clip.cond_stage_model.clip_l
        t5_model = clip.cond_stage_model.t5xxl

        # "Вытаскиваем" внутренние ТОКЕНИЗАТОРЫ
        clip_l_tokenizer = clip.tokenizer.clip_l
        t5_tokenizer = clip.tokenizer.t5xxl

        # Кодируем короткий промпт для CLIP-L
        tokens_l = clip_l_tokenizer.tokenize_with_weights(prompt_for_clip_l, return_word_ids=False)
        cond_l, pooled_l = clip_l_model.encode_token_weights(tokens_l)

        # Кодируем длинный промпт для T5
        tokens_t5 = t5_tokenizer.tokenize_with_weights(prompt_for_t5, return_word_ids=False)
        cond_t5, pooled_t5 = t5_model.encode_token_weights(tokens_t5)

        # Проверяем на None перед конкатенацией
        if pooled_l is None or pooled_t5 is None:
            raise ValueError("Одна из моделей не вернула 'pooled_output'. Проверьте файлы моделей.")

        # Совмещаем "pooled" выходы
        final_pooled = torch.cat([pooled_l, pooled_t5], dim=-1)

        # Собираем финальный conditioning
        final_conditioning = [[cond_l, {"pooled_output": final_pooled}], [cond_t5, {"pooled_output": final_pooled}]]

        return (final_conditioning,)


# --- Регистрация нод в ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "GayratFluxDualLoader": GayratFluxDualLoader,
    "GayratFluxEncoder": GayratFluxEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GayratFluxDualLoader": "Flux Dual Loader (Gayrat)",
    "GayratFluxEncoder": "Flux Separate Encoder (Gayrat)",
}
