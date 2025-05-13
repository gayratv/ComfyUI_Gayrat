# ComfyUI_Custom_Nodes_Gayrat/__init__.py

from .GoogleTranslateNode.google_translate_node import GoogleTranslateCLIPTextEncodeNode, GoogleTranslateTextNode

# Список узлов, которые будут зарегистрированы в ComfyUI
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": GoogleTranslateCLIPTextEncodeNode,
    "GoogleTranslateText": GoogleTranslateTextNode,
}

# Описание узлов (опционально, используется для отображения в интерфейсе)
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": "Google Translate (CLIP Text Encode)",
    "GoogleTranslateText": "Google Translate (Text Only)",
}