from googletrans import Translator, LANGUAGES
# импорт из Comfy
from server import PromptServer

### =====  GoogleTranslate Nodes [googletrans module]  ===== ###
translator = Translator()

def translate(prompt, srcTrans=None, toTrans=None):
    if not srcTrans:
        srcTrans = "auto"

    if not toTrans:
        toTrans = "en"

    translate_text_prompt = ""
    if prompt and prompt.strip() != "":
        translate_text_prompt = translator.translate(
            prompt, src=srcTrans, dest=toTrans
        )

    return translate_text_prompt.text if hasattr(translate_text_prompt, "text") else ""


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
                "manual_translate": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "Gyarat"

    def translate_text(self, **kwargs):
        from_translate = kwargs.get("from_translate")
        to_translate = kwargs.get("to_translate")
        manual_translate = kwargs.get("manual_translate", False)
        text = kwargs.get("text")
        clip = kwargs.get("clip")

        text_tranlsated = (
            translate(text, from_translate, to_translate)
            if not manual_translate
            else text
        )
        tokens = clip.tokenize(text_tranlsated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text_tranlsated)


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
        manual_translate = kwargs.get("manual_translate", False)
        text = kwargs.get("text")

        text_tranlsated = (
            translate(text, from_translate, to_translate)
            if not manual_translate
            else text
        )
        return (text_tranlsated,)


### =====  GoogleTranslate Nodes [googletrans module] -> end ===== ###