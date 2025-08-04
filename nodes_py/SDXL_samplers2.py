# ──────────────────────────────────────────────────────────────────────────────
#  SdxlSamplerParams  – модифицированная версия для SDXL и SD1.5
#  • Использует стандартный KSampler
# ──────────────────────────────────────────────────────────────────────────────
import os

import logging
import random
import time

import torch
import comfy.model_base
import comfy.samplers

from nodes import KSampler, LoraLoader
from comfy_extras.nodes_latent import LatentBatch
from comfy.utils import ProgressBar

# путь к папке проекта (родитель папки Samplers)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# теперь fonts в корне проекта
FONTS_DIR = os.path.join(PROJECT_ROOT, "fonts")


def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c


def parse_string_to_list(s):
    elements = s.split(',')
    result = []

    def parse_number(s):
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            return 0

    def decimal_places(s):
        if '.' in s:
            return len(s.split('.')[1])
        return 0

    for element in elements:
        element = element.strip()
        if '...' in element:
            start, rest = element.split('...')
            end, step = rest.split('+')
            decimals = decimal_places(step)
            start = parse_number(start)
            end = parse_number(end)
            step = parse_number(step)
            current = start
            if (start > end and step > 0) or (start < end and step < 0):
                step = -step
            while current <= end:
                result.append(round(current, decimals))
                current += step
        else:
            result.append(round(parse_number(element), decimal_places(element)))

    return result


# Sampler parameters with internal scheduler helper
# ──────────────────────────────────────────────────────────────────────────────
class SdxlSamplerParams:
    """
    Перебор семплеров/планировщиков/сидов и пр. для генерации латент-изображений
    """

    # ── инициализация ────────────────────────────────────────────────────────
    def __init__(self):
        self.loraloader = None
        self.lora = (None, None)

    # ── описание входов ───────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("STRING",
                         {"default": "123, ?, 456\n# Список сидов через запятую. '?' - случайный.", "multiline": True}),
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

    RETURN_TYPES = ("LATENT", "SAMPLER_PARAMS")
    RETURN_NAMES = ("latent", "params")
    FUNCTION = "execute"
    CATEGORY = "Gayrat/sampling"

    # ── основной метод ───────────────────────────────────────────────────────
    def execute(
            self,
            model,
            positive,
            negative,
            latent_image,
            seed,
            sampler,
            scheduler,
            steps,
            guidance,
            denoise,
            loras=None,
    ):

        # сиды
        noise_seeds = [
                          random.randint(0, 999_999) if "?" in n else int(n)
                          for n in seed.replace("\n", ",").split(",")
                          if n.strip() != ""
                      ] or [random.randint(0, 999_999)]

        # список сэмплеров
        if sampler == "*":
            sampler_list = comfy.samplers.KSampler.SAMPLERS
        elif sampler.startswith("!"):
            excl = [s.strip("! ") for s in sampler.replace("\n", ",").split(",")]
            sampler_list = [s for s in comfy.samplers.KSampler.SAMPLERS if s not in excl]
        else:
            sampler_list = [
                               s.strip() for s in sampler.replace("\n", ",").split(",")
                               if s.strip() in comfy.samplers.KSampler.SAMPLERS
                           ] or ["euler"]

        # список планировщиков (без фильтрации по зарегистрированным)
        scheduler_list = [s.strip() for s in scheduler.replace("\n", ",").split(",")] or ["simple"]

        # числовые списки
        steps_list = parse_string_to_list(steps or "20")
        denoise_list = parse_string_to_list(denoise or "1.0")
        guidance_list = parse_string_to_list(guidance or "7.0")

        # conditioning
        if isinstance(positive, dict) and "encoded" in positive:
            cond_texts = positive["text"]
            positive_list = positive["encoded"]
        else:
            cond_texts = [None]
            positive_list = [positive]

        # подготовка хелперов
        sampler_node = KSampler()
        latentbatch = LatentBatch()

        width = latent_image["samples"].shape[3] * 8
        height = latent_image["samples"].shape[2] * 8

        # LORA
        lora_strength_len = 1
        if loras:
            lora_models = loras["loras"]
            lora_strengths = loras["strengths"]
            lora_strength_len = sum(len(lst) for lst in lora_strengths)
            self.loraloader = self.loraloader or LoraLoader()

        # прогресс-бар
        total = (
                len(positive_list) * len(noise_seeds)
                * len(guidance_list) * len(sampler_list) * len(scheduler_list)
                * len(steps_list) * len(denoise_list) * lora_strength_len
        )
        pbar = ProgressBar(total) if total > 1 else None

        # выходы
        out_latent = None
        out_params = []
        counter = 0

        # ──────────────────────────────────────────────────────────────────────
        #  перебор всех комбинаций
        # ──────────────────────────────────────────────────────────────────────
        for l_idx in range(lora_strength_len):
            patched_model = (
                self.loraloader.load_lora(
                    model, None, lora_models[0], lora_strengths[0][l_idx], 0
                )[0] if loras else model
            )

            for c_idx, pos_cond in enumerate(positive_list):
                prompt = cond_texts[c_idx] if cond_texts[0] else None

                for seed_val in noise_seeds:
                    for g in guidance_list:
                        for samp in sampler_list:
                            for sched in scheduler_list:
                                for st in steps_list:
                                    for dn in denoise_list:

                                        counter += 1
                                        logging.info(
                                            "Sample %d/%d | seed=%s sampler=%s "
                                            "scheduler=%s steps=%s guidance=%s "
                                            "denoise=%s%s",
                                            counter, total, seed_val, samp, sched,
                                            st, g, dn,
                                            f" lora={lora_models[0]} strength={lora_strengths[0][l_idx]}"
                                            if loras else ""
                                        )

                                        t0 = time.time()
                                        latent = sampler_node.sample(
                                            model=patched_model,
                                            seed=seed_val,
                                            steps=st,
                                            cfg=g,
                                            sampler_name=samp,
                                            scheduler=sched,
                                            denoise=dn,
                                            positive=pos_cond,
                                            negative=negative[0],
                                            latent_image=latent_image
                                        )[0]
                                        elapsed = time.time() - t0

                                        out_params.append({
                                            "time": elapsed,
                                            "seed": seed_val,
                                            "width": width,
                                            "height": height,
                                            "sampler": samp,
                                            "scheduler": sched,
                                            "steps": st,
                                            "guidance": g,
                                            "denoise": dn,
                                            "prompt": prompt,
                                            **({"lora": lora_models[0],
                                                "lora_strength": lora_strengths[0][l_idx]} if loras else {})
                                        })

                                        out_latent = (
                                            latent if out_latent is None
                                            else latentbatch.batch(out_latent, latent)[0]
                                        )

                                        if pbar:
                                            pbar.update(1)

        # Обратите внимание, что мы больше не возвращаем 'sigmas'
        return out_latent, out_params


# Register node classes
NODE_CLASS_MAPPINGS = {
    "SdxlSamplerParams": SdxlSamplerParams,
}

# Display names for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SdxlSamplerParams": "SdxlSamplerParams (Gayrat)"
}