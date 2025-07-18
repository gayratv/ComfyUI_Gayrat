
"""
ComfyUI node: Pixtral Translate EN
Uses the Mistral model "pixtral-large-latest" to translate arbitrary text to English.

Input:
    text (STRING): text to translate
    temperature (FLOAT): sampling temperature (default 0.3)

Output:
    translated_en (STRING): translated text in English

Environment variable MISTRAL_API_KEY must be set with a valid Mistral API key.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import Tuple

import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# Model constants
# -------------------------------------------------------------------------

MODEL_NAME = "pixtral-large-latest"
DEFAULT_MAX_TOKENS = 2048

# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------

async def _call_mistral(api_key: str, payload: dict) -> dict:
    """Minimal async wrapper around Mistral chat/completions endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://api.mistral.ai/v1/chat/completions"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

# -------------------------------------------------------------------------
# Node definition
# -------------------------------------------------------------------------

class ComfyUIPixtralTranslate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_en",)

    FUNCTION = "translate"
    CATEGORY = "Gayrat/Pixtral Large"

    # -----------------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------------
    def translate(
        self,
        text: str,
        temperature: float = 0.3,
    ) -> Tuple[str]:
        """Translate the input text to English using Pixtral."""
        try:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Environment variable MISTRAL_API_KEY is not set.")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert translator. Translate the user's text into idiomatic, professional English. "
                        "Return only the translated text, without explanations or markup."
                    ),
                },
                {"role": "user", "content": text},
            ]

            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "top_p": 1.0,
            }

            resp = asyncio.run(_call_mistral(api_key, payload))
            translated = resp["choices"][0]["message"]["content"]

            return (translated,)

        except Exception as e:
            logger.exception("Error in ComfyUIPixtralTranslate.translate")
            err = str(e)
            return (err,)

# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {"ComfyUIPixtralTranslate": ComfyUIPixtralTranslate}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUIPixtralTranslate": "Pixtral Translate EN"}
