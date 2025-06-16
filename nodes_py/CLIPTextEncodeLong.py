import torch


class CLIPTextEncodeLong:
    """
    Эта нода принимает длинный текстовый промпт и автоматически разбивает его на части (чанки).
    Она добавляет "композиционный промпт" в начало каждой части, чтобы сохранить общую
    структуру сцены и избежать генерации только портретов.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "composition_prompt": ("STRING", {
                    "multiline": True,
                    "default": "photorealistic, full body shot, wide angle...",
                    "tooltip": "ПОЛЕ КОМПОЗИЦИИ:\nСюда впишите ключевые слова для общей сцены.\nНапример: full body shot, wide angle, woman with a panther."
                }),
                "details_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Your long prompt with details here...",
                    "tooltip": "ПОЛЕ ДЕТАЛЕЙ:\nСюда впишите все остальные детали: описание внешности, одежды, окружения, стиля и т.д."
                }),
                "chunk_size_words": ("INT", {"default": 60, "min": 10, "max": 77, "step": 1,
                                             "tooltip": "Количество слов в одной части. Слишком большое значение может вызвать ошибку лимита токенов."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Gayrat/conditioning"

    def encode(self, clip, details_prompt, composition_prompt, chunk_size_words):
        """
        Кодирует заданный текст с использованием предоставленной модели CLIP.

        Args:
            clip: Модель CLIP для токенизации и кодирования.
            details_prompt: Детальный текстовый промпт для кодирования.
            composition_prompt: Промпт с общей композицией сцены.
            chunk_size_words: Количество слов в каждой части.

        Returns:
            Кортеж, содержащий один тензор CONDITIONING.
        """

        # Убираем лишние пробелы
        composition_prompt = composition_prompt.strip()
        words = details_prompt.split()

        if not words and not composition_prompt:
            return ([],)

        # Если детальный промпт пуст, используем только композиционный
        if not words:
            text_chunks = [composition_prompt]
        else:
            # Простое разбиение на чанки по N слов
            text_chunks = []
            for i in range(0, len(words), chunk_size_words):
                chunk_of_words = words[i:i + chunk_size_words]
                # Добавляем композиционный промпт в начало каждого чанка
                full_chunk_text = f"{composition_prompt} {' '.join(chunk_of_words)}".strip()
                text_chunks.append(full_chunk_text)

        # Список для хранения закодированных 'conditioning'.
        final_conditioning = []

        print(f"Long prompt was split into {len(text_chunks)} chunks by {chunk_size_words} words each.")

        for chunk in text_chunks:
            # print(f"Encoding chunk: {chunk}")
            tokens = clip.tokenize(chunk)
            cond = clip.encode_from_tokens_scheduled(tokens)
            final_conditioning.extend(cond)

        if not final_conditioning:
            return ([],)

        return (final_conditioning,)


# --- Регистрация ноды в ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeLong": CLIPTextEncodeLong
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeLong": "CLIP Text Encode (Long & Composition)"
}
