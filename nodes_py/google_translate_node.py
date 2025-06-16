from googletrans import Translator, LANGUAGES
import asyncio
# импорт из Comfy
from server import PromptServer

### =====  GoogleTranslate Nodes [googletrans module]  ===== ###
translator = Translator()


async def translate(prompt, srcTrans=None, toTrans=None):
    """
    Асинхронная функция для перевода текста с использованием Google Translate.

    :param prompt: Текст для перевода.
    :param srcTrans: Исходный язык (по умолчанию "auto").
    :param toTrans: Целевой язык (по умолчанию "en").
    :return: Переведенный текст или пустая строка в случае ошибки.
    """
    if not srcTrans:
        srcTrans = "auto"  # Автоматическое определение исходного языка
    if not toTrans:
        toTrans = "en"  # По умолчанию перевод на английский

    # Проверка, что текст не пустой
    if not prompt or prompt.strip() == "":
        return ""

    try:
        # Выполняем асинхронный перевод
        translate_result = await translator.translate(prompt, src=srcTrans, dest=toTrans)
        # Возвращаем переведенный текст
        return translate_result.text if hasattr(translate_result, "text") else ""
    except Exception as e:
        # Обработка ошибок (например, проблемы с подключением к Google Translate)
        print(f"Translation error: {e}")
        return ""


def run_async(coro):
    """
    Запускает асинхронную корутину в текущем или новом цикле событий.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Если цикл уже запущен, используем run_coroutine_threadsafe
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    else:
        # Если цикл не запущен, используем asyncio.run
        return loop.run_until_complete(coro)


class GoogleTranslateCLIPTextEncodeNode:

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "from_translate": (
                    ["auto"] + list(LANGUAGES.keys()),
                    {"default": "auto"},
                ),
                "to_translate": (list(LANGUAGES.keys()), {"default": "en"}),
                "text": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    FUNCTION = "translate_text"
    DESCRIPTION = "This is a node that translates the prompt into another language using Google Translate."
    CATEGORY = "Gayrat/translate"

    def translate_text(self, **kwargs):
        from_translate = kwargs.get("from_translate")
        to_translate = kwargs.get("to_translate")
        text = kwargs.get("text")
        clip = kwargs.get("clip")

        # Запускаем асинхронный перевод через run_async
        text_translated = run_async(translate(text, from_translate, to_translate))

        tokens = clip.tokenize(text_translated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text_translated)


class GoogleTranslateTextNode(GoogleTranslateCLIPTextEncodeNode):

    @classmethod
    def INPUT_TYPES(self):
        return_types = super().INPUT_TYPES()
        del return_types["required"]["clip"]
        return return_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "translate_text"

    CATEGORY = "Gayrat"

    def translate_text(self, **kwargs):
        from_translate = kwargs.get("from_translate")
        to_translate = kwargs.get("to_translate")
        text = kwargs.get("text")

        # Запускаем асинхронный перевод через run_async
        text_translated = run_async(translate(text, from_translate, to_translate))

        return (text_translated,)

### =====  GoogleTranslate Nodes [googletrans module] -> end ===== ###

NODE_CLASS_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": GoogleTranslateCLIPTextEncodeNode,
    "GoogleTranslateText": GoogleTranslateTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": "Google Translate (CLIP Text Encode)",
    "GoogleTranslateText": "Google Translate (Text Only)"
}