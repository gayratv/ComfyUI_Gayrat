import asyncio
from googletrans import Translator

# --- Вспомогательный код ---
translator = Translator()

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    else:
        return loop.run_until_complete(coro)

async def do_translation(text):
    if not text or not text.strip():
        return ""
    try:
        translate_result = await translator.translate(text, src="auto", dest="en")
        return translate_result.text if hasattr(translate_result, "text") else ""
    except Exception as e:
        print(f"Translation error: {e}")
        return f"Translation error: {e}"

# --- Узел ---
class GoogleTranslateNode2:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "text_to_translate": ("STRING", {"multiline": True, "default": ""}), } }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "execute"
    CATEGORY = "Gayrat/translate"

    def execute(self, text_to_translate):
        # --- Отладочный вывод в консоль сервера ---
        print("\n[GoogleTranslateNode2 Debug] ---> Запуск узла.")
        translated_text = run_async(do_translation(text_to_translate))
        print(f"[GoogleTranslateNode2 Debug] Результат перевода: '{translated_text[:100]}...'") # Выводим первые 100 символов

        return_dict = {"result": (translated_text,), "ui": {"text": [translated_text]}}
        print(f"[GoogleTranslateNode2 Debug] Возвращаю в UI: {return_dict}")
        print("[GoogleTranslateNode2 Debug] ---< Конец выполнения.\n")
        # --------------------------------------------

        return return_dict

# --- Узел ---
class GoogleTranslateNode2CLIPTextEncodeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required":
                     {
                         "text_to_translate": ("STRING", {"multiline": True, "default": ""}),
                         "clip": ("CLIP",),
                       },
                }

    RETURN_TYPES = ("CONDITIONING","STRING",)
    RETURN_NAMES = ("CONDITIONING","translated_text",)
    FUNCTION = "execute"
    CATEGORY = "Gayrat/translate"
    DESCRIPTION = "This is a node that translates the prompt into another language using Google Translate."

    # def execute(self, text_to_translate):
    def execute(self, **kwargs):
        text_to_translate = kwargs.get("text")
        clip = kwargs.get("clip")
        translated_text = run_async(do_translation(text_to_translate))

        tokens = clip.tokenize(translated_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        # CONDITIONIG
        # return ([[cond, {"pooled_output": pooled}]], translated_text)
        conditioning = [[cond, {"pooled_output": pooled}]]

        # return_dict = {"result": (translated_text,), "ui": {"text": [translated_text]}}
        # Возвращаем и CONDITIONING, и переведенный текст в соответствии с RETURN_TYPES
        return_dict = {"result": (conditioning, translated_text), "ui": {"text": [translated_text]}}

        return return_dict

# --- Регистрация узла ---
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateNode2": GoogleTranslateNode2,
    "GoogleTranslateNode2CLIPTextEncodeNode": GoogleTranslateNode2CLIPTextEncodeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateNode2": "Google Translate 2" ,
    "GoogleTranslateNode2CLIPTextEncodeNode": "Google Translate2 (CLIP Text Encode)"
}