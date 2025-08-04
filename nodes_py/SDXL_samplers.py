# sdxl_sampler_params.py
# ------------------------------------------------------------
# ComfyUI custom node — **SdxlSamplerParams**
# Parameter sweeper designed for Stable Diffusion XL (also works with SD 1.5).
# Iterates over seeds, samplers, schedulers, steps, guidance‑scale and
# denoise, supports positive & negative conditioning and optional LoRA patches.
# Requires the *comfy_extras* package.
# ------------------------------------------------------------
# Place this file in `ComfyUI/custom_nodes` and restart ComfyUI.

from __future__ import annotations

import logging
import random
import time
from copy import deepcopy
from typing import Any, Dict, List

import comfy.samplers
import comfy.model_base
from comfy.utils import ProgressBar

# -----------------------------------------------------------------------------
# comfy_extras imports (mandatory)
# -----------------------------------------------------------------------------
try:
    from comfy_extras.nodes_custom_sampler import (
        Noise_RandomNoise,
        BasicGuider,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_latent import LatentBatch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "SdxlSamplerParams requires the *comfy_extras* package.\n"
        "Install it with:  pip install comfy_extras"
    ) from e

# -----------------------------------------------------------------------------
# Optional LoRA loader (also from comfy_extras)
# -----------------------------------------------------------------------------
try:
    from nodes import LoraLoader
except ImportError:  # pragma: no cover
    class LoraLoader:  # type: ignore
        def load_lora(self, *_, **__):
            raise RuntimeError("LoRA support unavailable – install comfy_extras.")

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _parse_numeric_list(spec: str) -> List[float | int]:
    """Parse flexible numeric list specifications: '30', '1,2,3', '4...10+2'."""
    spec = spec.strip()
    if spec == "":
        return []

    def _to_number(tok: str) -> float | int:
        return float(tok) if "." in tok else int(tok)

    vals: List[float | int] = []
    for part in spec.replace("\n", ",").split(","):
        part = part.strip()
        if part == "":
            continue
        if "..." in part:  # range syntax
            start_s, rest = part.split("...")
            end_s, step_s = rest.split("+")
            start, end, step = map(_to_number, (start_s, end_s, step_s))
            if (start < end and step <= 0) or (start > end and step >= 0):
                step = -step
            x = start
            while (x <= end) if step > 0 else (x >= end):
                vals.append(x)
                x += step
        else:
            vals.append(_to_number(part))
    return vals or [_to_number(spec)]


def _make_empty_negative(pos_cond):
    """Create unconditional stub conditioning with the same *structure* as positive."""
    try:
        stub = deepcopy(pos_cond)
        # If list of [text, extra] pairs – zero weights & texts
        if isinstance(stub, list):
            for item in stub:
                if isinstance(item, list) and len(item) == 2:
                    item[0] = ""
                    if isinstance(item[1], dict):
                        item[1]["weight"] = 0.0
        # CLIPConditioningData (SDXL) supports .multiply
        elif hasattr(stub, "multiply"):
            stub = stub.multiply(0.0)
        return stub
    except Exception:
        pass
    return pos_cond  # fallback same object

# -----------------------------------------------------------------------------
# Node class implementation
# -----------------------------------------------------------------------------

class SdxlSamplerParams:
    """Parameter sweeper node for SDXL & SD 1.5 models."""

    CATEGORY = "Gayrat/sampling"
    RETURN_TYPES = ("LATENT", "SAMPLER_PARAMS", "SIGMAS")
    RETURN_NAMES = ("latent", "params", "sigmas")
    FUNCTION = "execute"

    # ---------------------- ComfyUI INPUT_TYPES
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("STRING", {"default": "?", "multiline": True}),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "steps": ("STRING", {"default": "30"}),
                "guidance": ("STRING", {"default": "7.0"}),
                "denoise": ("STRING", {"default": "1.0"}),
            },
            "optional": {"loras": ("LORA_PARAMS",)},
        }

    # ---------------------- sigma schedule helper
    @staticmethod
    def _sigmas(model, scheduler: str, steps: int, denoise: float):
        total = steps if denoise == 1 else int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total
        ).cpu()
        return sigmas[-(steps + 1) :]

    # ---------------------- init
    def __init__(self):
        self._loraloader: LoraLoader | None = None

    # ---------------------- main execute logic
    def execute(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed: str,
        sampler: str,
        scheduler: str,
        steps: str,
        guidance: str,
        denoise: str,
        loras=None,
    ):
        # -- Model type check (skip on older builds)
        try:
            if model.model.model_type != comfy.model_base.ModelType.DIFFUSION:
                raise ValueError("Only diffusion models (SDXL / SD1.5) supported.")
        except AttributeError:
            pass  # enum missing in old versions – ignore

        # -- Ensure negative conditioning exists
        if negative is None:
            logging.info("Negative conditioning missing – using empty stub.")
            negative = _make_empty_negative(positive)

        # -- Build iteration lists
        seed_vals = [
            random.randint(0, 2**32 - 1) if "?" in s else int(s)
            for s in seed.replace("\n", ",").split(",")
            if s.strip()
        ] or [random.randint(0, 2**32 - 1)]

        all_samplers = comfy.samplers.KSampler.SAMPLERS
        sampler = sampler.strip()
        if sampler == "*":
            sampler_vals = all_samplers
        elif sampler.startswith("!"):
            excl = {x.strip().lstrip("!") for x in sampler.split(",")}
            sampler_vals = [s for s in all_samplers if s not in excl]
        else:
            sampler_vals = [s.strip() for s in sampler.split(",") if s.strip()] or ["euler"]

        sched_vals = [s.strip() for s in scheduler.split(",") if s.strip()] or ["simple"]
        steps_vals = _parse_numeric_list(steps)
        guide_vals = _parse_numeric_list(guidance)
        dn_vals = _parse_numeric_list(denoise)

        # -- LoRA handling
        if loras:
            self._loraloader = self._loraloader or LoraLoader()
            lora_models = loras["loras"]
            lora_strengths = loras["strengths"]
            lora_total = sum(len(lst) for lst in lora_strengths)
        else:
            lora_total = 1

        # -- Helpers
        guider_factory = BasicGuider()
        sampler_adv = SamplerCustomAdvanced()
        latent_batcher = LatentBatch()

        # -- Progress bar
        total = (
            len(seed_vals)
            * len(sampler_vals)
            * len(sched_vals)
            * len(steps_vals)
            * len(guide_vals)
            * len(dn_vals)
            * lora_total
        )
        pbar = ProgressBar(total) if total > 1 else None

        latents_batched = None
        params_log: List[Dict[str, Any]] = []
        last_sigmas = None

        # -- Nested sweep
        for l_idx in range(lora_total):
            current_model = model
            if loras:
                current_model, _ = self._loraloader.load_lora(
                    model, None, lora_models[0], lora_strengths[0][l_idx], 0
                )

            for s in seed_vals:
                noise_node = Noise_RandomNoise(s)

                for cfg in guide_vals:
                                        # --- DEBUG: log incoming conditioning types
                    logging.warning(
                        "[SdxlSamplerParams] positive type=%s | negative type=%s",
                        type(positive),
                        type(negative),
                    )
                    # If mistakenly passed raw text, wrap into simple conditioning stub
                    if isinstance(positive, str):
                        logging.error("positive came as str – wrapping into stub conditioning!")
                        positive = [[positive, {"weight": 1.0}]]
                    if isinstance(negative, str):
                        logging.error("negative came as str – wrapping into stub conditioning!")
                        negative = [[negative, {"weight": 0.0}]]
                    cond_dict = {
                        "positive": positive,
                        "negative": negative,
                        "cond_scale": cfg,
                    }
                    guider = guider_factory.get_guider(current_model, cond_dict)[0]

                    for samp_name in sampler_vals:
                        samp_obj = comfy.samplers.sampler_object(samp_name)

                        for sched_name in sched_vals:
                            for st in steps_vals:
                                for dn in dn_vals:
                                    sigmas = self._sigmas(current_model, sched_name, int(st), float(dn))
                                    last_sigmas = sigmas

                                    t0 = time.time()
                                    latent = sampler_adv.sample(
                                        noise_node, guider, samp_obj, sigmas, latent_image
                                    )[1]
                                    elapsed = time.time() - t0

                                    record = {
                                        "seed": s,
                                        "sampler": samp_name,
                                        "scheduler": sched_name,
                                        "steps": st,
                                        "guidance": cfg,
                                        "denoise": dn,
                                        "time": round(elapsed, 3),
                                    }
                                    if loras:
                                        record.update(
                                            {
                                                "lora": lora_models[0],
                                                "lora_strength": lora_strengths[0][l_idx],
                                            }
                                        )
                                    params_log.append(record)

                                    latents_batched = (
                                        latent if latents_batched is None else latent_batcher.batch(latents_batched, latent)[0]
                                    )

                                    if pbar:
                                        pbar.update(1)

        return latents_batched, params_log, last_sigmas


# -----------------------------------------------------------------------------
# Registration dictionaries (ComfyUI)
# -----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"SdxlSamplerParams": SdxlSamplerParams}
NODE_DISPLAY_NAME_MAPPINGS = {"SdxlSamplerParams": "SdxlSamplerParams (Gayrat)"}
