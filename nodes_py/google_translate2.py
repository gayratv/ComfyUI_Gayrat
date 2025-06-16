import asyncio
from googletrans import Translator

# --- Вспомогательный код (без изменений) ---

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


# --- Наш узел ---

class GoogleTranslateNode2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_to_translate": ("STRING", {"multiline": True, "default": "A beautiful cat in a fantasy world"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "execute"

    # ИЗМЕНЕНИЕ: Категория узла обновлена
    CATEGORY = "Gayrat/translate"

    def execute(self, text_to_translate):
        translated_text = run_async(
            do_translation(text_to_translate)
        )
        return {"result": (translated_text,), "ui": {"translated_text": [translated_text]}}


# --- Регистрация узла ---
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateNode2": GoogleTranslateNode2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateNode2": "Google Translate 2"
}