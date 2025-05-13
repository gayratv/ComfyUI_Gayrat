# ComfyUI_Gayrat/__init__.py

from .GoogleTranslateNode.google_translate_node import GoogleTranslateCLIPTextEncodeNode, GoogleTranslateTextNode
from .ImageResize.image_scale_by_aspect_ratio_v2 import NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_ImageResize
from .ImageResize.image_scale_by_aspect_ratio_v2 import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_ImageResize

# Список узлов, которые будут зарегистрированы в ComfyUI
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": GoogleTranslateCLIPTextEncodeNode,
    "GoogleTranslateText": GoogleTranslateTextNode,
}
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_ImageResize)

# ==========================================
# Описание узлов (опционально, используется для отображения в интерфейсе)
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": "Google Translate (CLIP Text Encode)",
    "GoogleTranslateText": "Google Translate (Text Only)",
}

NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_ImageResize)