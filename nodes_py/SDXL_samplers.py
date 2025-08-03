"""
Файл - копия K-sampler для перебора samplers

diffusion_sampler_params.py

ComfyUI custom node “DiffusionSamplerParams”

Iterates over seeds, samplers, schedulers, steps, guidance‑scale and denoise
values to generate batches of latents for *diffusion* models (SD 1.5, SDXL).

Inspired by FluxSamplerParams but streamlined: no max/base shift logic,
no FLUX / FLOW patching – it works directly with ModelType.DIFFUSION.

Place this file in your ComfyUI `custom_nodes` folder.
"""

import logging
import random
import time
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.model_base
from comfy.utils import ProgressBar

# Optional extras (present in comfy_extras); we fall back gracefully
try:
    from comfy_extras.nodes_custom_sampler import (
        Noise_RandomNoise,
        BasicGuider,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_latent import LatentBatch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "DiffusionSamplerParams requires the comfy_extras package "
        "(`pip install comfy_extras`)."
    ) from e

try:
    from nodes import LoraLoader
except ImportError:
    # If comfy_extras is available LoraLoader will be there; otherwise stub.
    class LoraLoader:  # type: ignore
        def load_lora(self, *args, **kwargs):
            raise RuntimeError("LoRA support unavailable – install comfy_extras.")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def parse_string_to_list(s: str) -> List[float | int]:
    """
    Parses a flexible numeric list specification.

    Examples
    --------
    ``"1, 2, 3"`` → [1, 2, 3]
    ``"4...10+2"`` → [4, 6, 8, 10]

    Returns ints when possible, floats otherwise.
    """
    elements = s.split(",")
    result: List[float | int] = []

    def _parse_number(token: str) -> float | int:
        try:
            return float(token) if "." in token else int(token)
        except ValueError:
            return 0

    def _decimal_places(token: str) -> int:
        return len(token.split(".")[1]) if "." in token else 0

    for elem in elements:
        elem = elem.strip()
        if "..." in elem:  # range syntax  start...end+step
            start, rest = elem.split("...")
            end, step = rest.split("+")
            decimals = _decimal_places(step)
            start_n, end_n, step_n = map(_parse_number, (start, end, step))
            # direction agnostic
            if (start_n > end_n and step_n > 0) or (start_n < end_n and step_n < 0):
                step_n = -step_n
            current = start_n
            while current <= end_n if step_n > 0 else current >= end_n:
                result.append(round(current, decimals))
                current += step_n
        else:
            value = _parse_number(elem)
            if isinstance(value, float):
                value = round(value, _decimal_places(elem))
            result.append(value)
    return result


class SdxlSamplerParams:
    """
    Parameter sweeper for ComfyUI diffusion models (SD 1.5, SDXL).

    Returns
    -------
    LATENT : torch.Tensor
        Batched latent samples (B, C, H/8, W/8).
    SAMPLER_PARAMS : List[Dict[str, Any]]
        Parameter dictionary for each sample.
    SIGMAS : torch.FloatTensor
        Sigma schedule used for the *last* sample (steps + 1 values).
    """

    # ────────────────────────────────────────────────────────────────────
    # Static helpers
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_sigmas(
        model,
        scheduler: str,
        steps: int,
        denoise: float,
    ) -> torch.FloatTensor:
        """
        Compute a sigma schedule identical to KSampler behaviour.
        """
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return torch.FloatTensor([])
            total_steps = int(steps / denoise)

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps
        ).cpu()
        return sigmas[-(steps + 1) :]

    # ────────────────────────────────────────────────────────────────────
    # ComfyUI node interface
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "STRING",
                    {
                        "default": "?, 123, 456\n# Comma‑separated list. '?' – random.",
                        "multiline": True,
                    },
                ),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "steps": ("STRING", {"default": "20"}),
                "guidance": ("STRING", {"default": "7.0"}),
                "denoise": ("STRING", {"default": "1.0"}),
            },
            "optional": {
                "loras": ("LORA_PARAMS",),
            },
        }

    RETURN_TYPES = ("LATENT", "SAMPLER_PARAMS", "SIGMAS")
    RETURN_NAMES = ("latent", "params", "sigmas")
    FUNCTION = "execute"
    CATEGORY = "Gayrat/sampling"

    # ────────────────────────────────────────────────────────────────────
    # Core execution
    # ────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.loraloader: LoraLoader | None = None

    def execute(
        self,
        model,
        conditioning,
        latent_image,
        seed: str,
        sampler: str,
        scheduler: str,
        steps: str,
        guidance: str,
        denoise: str,
        loras=None,
    ):
        if model.model.model_type != comfy.model_base.ModelType.DIFFUSION:
            raise ValueError("DiffusionSamplerParams requires a DIFFUSION model.")

        # Seeds ────────────────────────────────────────────────────────
        noise_seeds = [
            random.randint(0, 999_999) if "?" in n else int(n)
            for n in seed.replace("\n", ",").split(",")
            if n.strip()
        ] or [random.randint(0, 999_999)]

        # Samplers ─────────────────────────────────────────────────────
        if sampler.strip() == "*":
            sampler_list = comfy.samplers.KSampler.SAMPLERS
        elif sampler.strip().startswith("!"):
            excluded = {s.strip("! ") for s in sampler.replace("\n", ",").split(",")}
            sampler_list = [s for s in comfy.samplers.KSampler.SAMPLERS if s not in excluded]
        else:
            sampler_list = [
                s.strip()
                for s in sampler.replace("\n", ",").split(",")
                if s.strip() in comfy.samplers.KSampler.SAMPLERS
            ] or ["euler"]

        # Schedulers (we allow any custom string) ──────────────────────
        scheduler_list = [s.strip() for s in scheduler.replace("\n", ",").split(",")] or ["simple"]

        # Numeric lists ────────────────────────────────────────────────
        steps_list = parse_string_to_list(steps or "20")
        guidance_list = parse_string_to_list(guidance or "7.0")
        denoise_list = parse_string_to_list(denoise or "1.0")

        # Conditioning (batch‑compatible) ──────────────────────────────
        if isinstance(conditioning, dict) and "encoded" in conditioning:
            cond_texts = conditioning["text"]
            cond_enc = conditioning["encoded"]
        else:
            cond_texts = [None]
            cond_enc = [conditioning]

        # LoRA setup ───────────────────────────────────────────────────
        lora_strength_len = 1
        if loras:
            lora_models = loras["loras"]
            lora_strengths = loras["strengths"]
            lora_strength_len = sum(len(lst) for lst in lora_strengths)
            self.loraloader = self.loraloader or LoraLoader()

        # Helpers ──────────────────────────────────────────────────────
        basicguider = BasicGuider()
        sampler_adv = SamplerCustomAdvanced()
        latentbatch = LatentBatch()

        width = latent_image["samples"].shape[3] * 8
        height = latent_image["samples"].shape[2] * 8

        # Progress bar ─────────────────────────────────────────────────
        total = (
            len(cond_enc)
            * len(noise_seeds)
            * len(guidance_list)
            * len(sampler_list)
            * len(scheduler_list)
            * len(steps_list)
            * len(denoise_list)
            * lora_strength_len
        )
        pbar = ProgressBar(total) if total > 1 else None

        # Outputs ──────────────────────────────────────────────────────
        out_latent = None
        out_params: List[Dict[str, Any]] = []
        out_sigmas = None
        counter = 0

        # Sweep all combinations ───────────────────────────────────────
        for l_idx in range(lora_strength_len):
            patched_model = (
                self.loraloader.load_lora(
                    model, None, lora_models[0], lora_strengths[0][l_idx], 0
                )[0]
                if loras
                else model
            )

            for c_idx, cond in enumerate(cond_enc):
                prompt = cond_texts[c_idx] if cond_texts[0] else None

                for seed_val in noise_seeds:
                    noise_node = Noise_RandomNoise(seed_val)

                    for guidance_val in guidance_list:
                        cond_val = [
                            t if isinstance(t, list) else t.copy()
                            for t in cond
                        ] if isinstance(cond, list) else cond
                        # Inject guidance into conditioning dict
                        cond_val = [
                            [t[0], {**t[1], "guidance": guidance_val}] for t in cond_val
                        ] if isinstance(cond_val, list) else cond_val

                        guider = basicguider.get_guider(patched_model, cond_val)[0]

                        for samp in sampler_list:
                            samp_obj = comfy.samplers.sampler_object(samp)

                            for sched in scheduler_list:
                                for st in steps_list:
                                    for dn in denoise_list:
                                        sigmas = self._get_sigmas(patched_model, sched, st, dn)
                                        out_sigmas = sigmas

                                        counter += 1
                                        logging.info(
                                            "Sample %d/%d | seed=%s sampler=%s "
                                            "scheduler=%s steps=%s guidance=%s denoise=%s%s",
                                            counter,
                                            total,
                                            seed_val,
                                            samp,
                                            sched,
                                            st,
                                            guidance_val,
                                            dn,
                                            f" lora={lora_models[0]} strength={lora_strengths[0][l_idx]}"
                                            if loras
                                            else "",
                                        )

                                        t0 = time.time()
                                        latent = sampler_adv.sample(
                                            noise_node, guider, samp_obj, sigmas, latent_image
                                        )[1]
                                        elapsed = time.time() - t0

                                        out_params.append(
                                            {
                                                "time": elapsed,
                                                "seed": seed_val,
                                                "width": width,
                                                "height": height,
                                                "sampler": samp,
                                                "scheduler": sched,
                                                "steps": st,
                                                "guidance": guidance_val,
                                                "denoise": dn,
                                                "prompt": prompt,
                                                **(
                                                    {
                                                        "lora": lora_models[0],
                                                        "lora_strength": lora_strengths[0][l_idx],
                                                    }
                                                    if loras
                                                    else {}
                                                ),
                                            }
                                        )

                                        out_latent = (
                                            latent
                                            if out_latent is None
                                            else latentbatch.batch(out_latent, latent)[0]
                                        )

                                        if pbar:
                                            pbar.update(1)

        return out_latent, out_params, out_sigmas


# ────────────────────────────────────────────────────────────────────────────
# Node registration
# ────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "SdxlSamplerParams": SdxlSamplerParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SdxlSamplerParams": "SDXL/SD1.5 Sampler Params",
}
