"""ComfyUI node: Pixtral Large (Mistral multimodal)
Одномодельная версия — *pixtral‑large‑latest*.

* **Изображение опционально**: без него узел ведёт себя как обычный LLM‑чат.
* После получения первого ответа текст переводится на английский тем же API и возвращается вторым выходом.
* Ключ читается из окружения **MISTRAL_API_KEY**.
* `max_tokens` фиксирован (2048 ≤ 8192), `top_p` = 0.85 по умолчанию.
"""

import base64
import io
import logging
import asyncio
import os
from typing import Optional, Tuple

import httpx
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Константы модели
# -----------------------------------------------------------------------------

MODEL_NAME = "pixtral-large-latest"
MAX_TOKENS_LIMIT = 8192
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.85

# -----------------------------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------------------------

def _tensor_to_pil(img_tensor: torch.Tensor | "np.ndarray") -> Image.Image:
    """CHW/HWC → RGB PIL (float 0‑1). Альфа отбрасывается."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.squeeze().cpu().numpy()
    else:
        img = img_tensor

    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)

    if img.ndim == 2:
        return Image.fromarray((img * 255).astype("uint8"), "L").convert("RGB")
    if img.ndim == 3:
        mode = "RGBA" if img.shape[2] == 4 else "RGB"
        return Image.fromarray((img * 255).astype("uint8"), mode).convert("RGB")
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _pil_to_data_url(pil: Image.Image, *, quality: int = 85, max_size: int = 1024) -> str:
    w, h = pil.size
    scale = max(w, h) / max_size
    if scale > 1:
        pil = pil.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


async def _call_mistral(api_key: str, payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://api.mistral.ai/v1/chat/completions"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


# -----------------------------------------------------------------------------
# Определение ноды
# -----------------------------------------------------------------------------

class ComfyUIPixtralLarge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Describe the image"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.1}),
                "top_p": ("FLOAT", {"default": DEFAULT_TOP_P, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_original", "response_en")

    FUNCTION = "process"
    CATEGORY = "Gayrat/Pixtral Large"

    # ------------------------------------------------------------------
    # Основная логика
    # ------------------------------------------------------------------

    def process(
        self,
        prompt: str,
        temperature: float,
        top_p: float = DEFAULT_TOP_P,
        image: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str]:
        try:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Environment variable MISTRAL_API_KEY is not set.")

            # ---------- формируем первое сообщение ----------
            content = [{"type": "text", "text": prompt}]
            if image is not None:
                img_url = _pil_to_data_url(_tensor_to_pil(image))
                content.append({"type": "image_url", "image_url": {"url": img_url}})

            payload1 = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "top_p": top_p,
            }

            # ---------- вызываем Mistral для первого ответа ----------
            first_resp = asyncio.run(_call_mistral(api_key, payload1))
            original_text = first_resp["choices"][0]["message"]["content"]

            # ---------- формируем запрос на перевод ----------
            translate_messages = [
                {"role": "system", "content": "Translate the following text to English. Respond with only the translation."},
                {"role": "user", "content": original_text},
            ]
            payload2 = {
                "model": MODEL_NAME,
                "messages": translate_messages,
                "temperature": 0.3,
                "max_tokens": DEFAULT_MAX_TOKENS // 2,
                "top_p": 1.0,
            }

            translate_resp = asyncio.run(_call_mistral(api_key, payload2))
            translated_text = translate_resp["choices"][0]["message"]["content"]

            return (original_text, translated_text)

        except Exception as e:
            logger.exception("Error in ComfyUIPixtralLarge.process")
            err = str(e)
            return (err, err)


# -----------------------------------------------------------------------------
# Регистрация
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {"ComfyUIPixtralLarge": ComfyUIPixtralLarge}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUIPixtralLarge": "Pixtral Large"}
