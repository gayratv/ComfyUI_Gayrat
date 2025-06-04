# ComfyUI_Gayrat/__init__.py

from .GoogleTranslateNode.google_translate_node import GoogleTranslateCLIPTextEncodeNode, GoogleTranslateTextNode
from .ImageResize.image_scale_by_aspect_ratio_v2 import ImageScaleByAspectRatioV2
from .ergouzi.EGJDFDHT import EGRYHT
from .MaskAreaCondition.adaptive_crop_stitch_params_node import AdaptiveParamsWithModelChoice
from .SaveImageAndMask.save_image_with_mask import SaveImageWithMask

# Список узлов, которые будут зарегистрированы в ComfyUI
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": GoogleTranslateCLIPTextEncodeNode,
    "GoogleTranslateText": GoogleTranslateTextNode,
    "ImageResize: ImageScaleByAspectRatio V2": ImageScaleByAspectRatioV2,
    "EG_RY_HT" : EGRYHT,
    "AdaptiveParamsWithModelChoice": AdaptiveParamsWithModelChoice,
    "SaveImageWithMask": SaveImageWithMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateCLIPTextEncode": "Google Translate (CLIP Text Encode)",
    "GoogleTranslateText": "Google Translate (Text Only)",
    "ImageResize": "ImageScaleByAspectRatio V2",
    "EG_RY_HT": "Float slider",
    "AdaptiveParamsWithModelChoice": "Adaptive Params (Model Choice + Passthrough)",
    "SaveImageWithMask": "Save Image with Mask"
}
