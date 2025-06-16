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
        translated_text = run_async(do_translation(text_to_translate))

        return_dict = {"result": (translated_text,), "ui": {"text": [translated_text]}}

        return return_dict

# --- Узел ---
class GoogleTranslateNode2CLIPTextEncodeNode:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "text_to_translate": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = ("CONDITIONING","translated_text",)
    FUNCTION = "execute"
    CATEGORY = "Gayrat/translate"
    DESCRIPTION = "This is a node that translates the prompt into another language using Google Translate."

    def execute(self, **kwargs):
        text_to_translate = kwargs.get("text_to_translate")
        clip = kwargs.get("clip")
        translated_text = run_async(do_translation(text_to_translate))

        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        # tokens = clip.tokenize(translated_text)
        # cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        #
        #
        # conditioning = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(translated_text)
        # conditioning= (clip.encode_from_tokens_scheduled(tokens),)
        conditioning= clip.encode_from_tokens_scheduled(tokens)

        return_dict = {"result": (conditioning, translated_text), "ui": {"text": [translated_text]}}
        return return_dict

        # return (conditioning, translated_text)

        # return ([[cond, {"pooled_output": pooled}]], translated_text)


# --- Регистрация узла ---
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateNode2": GoogleTranslateNode2,
    "GoogleTranslateNode2CLIPTextEncodeNode": GoogleTranslateNode2CLIPTextEncodeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateNode2": "Google Translate 2" ,
    "GoogleTranslateNode2CLIPTextEncodeNode": "Google Translate2 (CLIP Text Encode)"
}