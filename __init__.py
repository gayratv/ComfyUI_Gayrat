# ComfyUI_Gayrat/__init__.py

from .GoogleTranslateNode.google_translate_node import (
    GoogleTranslateCLIPTextEncodeNode,
    GoogleTranslateTextNode
)
from .ImageResize.image_scale_by_aspect_ratio_v2 import ImageScaleByAspectRatioV2
from .ergouzi.EGJDFDHT import EGRYHT
from .MaskAreaCondition.adaptive_crop_stitch_params_node import AdaptiveParamsWithModelChoice
from .SaveImageAndMask.save_image_with_alpha import SaveImageWithAlpha
from .SaveImageAndMask.LoadImageWithTrimOptions import LoadImageWithTrimOptions
from .SaveImageAndMask.сustom_load_image import CustomLoadImage

# Импорт пользовательских нод из Samplers/cust_samplers.py
from .Samplers.cust_samplers import (
    NODE_CLASS_MAPPINGS as CUST_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CUST_NODE_DISPLAY_NAME_MAPPINGS
)

# Список узлов, которые будут зарегистрированы в ComfyUI
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": GoogleTranslateCLIPTextEncodeNode,
    "GoogleTranslateText": GoogleTranslateTextNode,
    "ImageResize: ImageScaleByAspectRatio V2": ImageScaleByAspectRatioV2,
    "EG_RY_HT": EGRYHT,
    "AdaptiveParamsWithModelChoice": AdaptiveParamsWithModelChoice,
    "SaveImageWithAlpha": SaveImageWithAlpha,
    "LoadImageWithTrimOptions": LoadImageWithTrimOptions,
    "LoadImageCustomFromFileWithSizeCorrectMask": CustomLoadImage,
}

# Обновляем основную карту узлов пользовательскими
NODE_CLASS_MAPPINGS.update(CUST_NODE_CLASS_MAPPINGS)

# Отображаемые имена узлов в UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": "Google Translate (CLIP Text Encode)",
    "GoogleTranslateText": "Google Translate (Text Only)",
    "ImageResize": "ImageScaleByAspectRatio V2",
    "EG_RY_HT": "Float slider",
    "AdaptiveParamsWithModelChoice": "Adaptive Params (Model Choice + Passthrough)",
    "SaveImageWithAlpha": "Save Image with Alpha",
    "LoadImageWithTrimOptions": "Load Image (Trim + Full)",
    "LoadImageCustomFromFileWithSizeCorrectMask": "Load Image (Custom, Correct Mask Size)",
}

# Обновляем отображаемые имена узлов пользовательскими
NODE_DISPLAY_NAME_MAPPINGS.update(CUST_NODE_DISPLAY_NAME_MAPPINGS)
