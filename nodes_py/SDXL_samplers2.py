# ──────────────────────────────────────────────────────────────────────────────
#  FluxSamplerParams  – модифицированная версия
#  • Поддерживает кастомные планировщики (например, OptimalStepsScheduler)
#  • Для имени плaнировщика "optimal"/"optimal_steps"/"optimalsteps" вызывает
#    узел OptimalStepsScheduler, которому передаётся model_type="FLUX".
#  • В остальных случаях использует comfy.samplers.calculate_sigmas, как прежде
# ──────────────────────────────────────────────────────────────────────────────
import os

import logging
import random
import time

import torch
import comfy.model_base
import comfy.samplers

from comfy_extras.nodes_optimalsteps import OptimalStepsScheduler
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
# from node_helpers import conditioning_set_values, parse_string_to_list
from nodes import LoraLoader
from comfy.utils import ProgressBar

# # ──   узлы/утилиты, которые уже есть в ComfyUI  ───────────────────────────────
# from nodes import (
#     # BasicGuider,
#     SamplerCustomAdvanced,
#     LatentBatch,
#     ModelSamplingFlux,
#     ModelSamplingAuraFlow,
#     Noise_RandomNoise,
#     ProgressBar,
#     OptimalStepsScheduler,       # ← наш «особый» узел
#     LoraLoader,
# )
#
# from utils import (
#     parse_string_to_list,
#     conditioning_set_values,
# )


import torch.nn.functional as F

import torchvision.transforms.v2 as T
# import torchvision
# T = torchvision.transforms.v2



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


# Flux sampler parameters with internal scheduler helper
# ──────────────────────────────────────────────────────────────────────────────
class SdxlSamplerParams:
    """
    Перебор семплеров/планировщиков/сидов и пр. для генерации латент-изображений
    """

    # ── статический расчёт сигм ──────────────────────────────────────────────
    @staticmethod
    def _get_sigmas(model, scheduler: str, steps: int, denoise: float):
        """
        Возвращает torch.FloatTensor длиной steps+1
        """


        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return torch.FloatTensor([])
            total_steps = int(steps / denoise)

        sched_l = scheduler.lower()
        if sched_l in {"optimal", "optimal_steps", "optimalsteps","optimalstepsscheduler"}:
            sig_out = OptimalStepsScheduler().get_sigmas("FLUX", total_steps, denoise)
            sigmas = sig_out[0] if isinstance(sig_out, (tuple, list)) else sig_out
            sigmas = sigmas.cpu()
        else:
            sigmas = comfy.samplers.calculate_sigmas(
                model.get_model_object("model_sampling"), scheduler, total_steps
            ).cpu()

        return sigmas[-(steps + 1):]

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
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                # "seed": ("STRING", {"default": "?"}),
                # ИЗМЕНЕНИЕ: Тип поля изменен на STRING с multiline, и добавлена подсказка в default
                "seed": ("STRING",
                         {"default": "123, ?, 456\n# Список сидов через запятую. '?' - случайный.", "multiline": True}),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "steps": ("STRING", {"default": "20"}),
                "guidance": ("STRING", {"default": "3.5"}),
                "max_shift": ("STRING", {"default": "1.15"}),
                "base_shift": ("STRING", {"default": "0.5"}),
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

    # ── основной метод ───────────────────────────────────────────────────────
    def execute(
        self,
        model,
        conditioning,
        latent_image,
        seed,
        sampler,
        scheduler,
        steps,
        guidance,
        max_shift,
        base_shift,
        denoise,
        loras=None,
    ):
        is_flow = model.model.model_type    == comfy.model_base.ModelType.FLOW
        is_flux_model = model.model.model_type == comfy.model_base.ModelType.FLUX

        # Переменная для хранения результата
        is_schnell_model = False

        # 1. Сначала проверяем, что это в принципе модель типа FLUX
        if model.model.model_type == comfy.model_base.ModelType.FLUX:
            # 2. Затем проверяем, что у модели есть конфигурация и имя
            #    Это безопасный способ избежать ошибок, если атрибута нет
            if hasattr(model, 'model_config') and hasattr(model.model_config, 'name'):
                # 3. Приводим имя к нижнему регистру и ищем подстроку "schnell"
                if 'schnell' in model.model_config.name.lower():
                    is_schnell_model = True

        # Теперь переменная is_schnell_model будет True только для моделей
        # типа FLUX, в названии которых есть "schnell", например, "FLUX.1-schnell.safetensors"

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
        steps_list     = parse_string_to_list(steps     or ("4"  if is_flow else "20"))
        denoise_list   = parse_string_to_list(denoise   or "1.0")
        guidance_list  = parse_string_to_list(guidance  or "3.5")
        max_shift_list = parse_string_to_list(max_shift or ("0"  if is_flow else "1.15"))

        # if not is_schnell:
        #     max_shift = "1.15" if max_shift == "" else max_shift
        #     base_shift = "0.5" if base_shift == "" else base_shift
        # else:
        #     max_shift = "0"
        #     base_shift = "1.0" if base_shift == "" else base_shift
        if is_schnell_model:
          max_shift = "0"
          base_shift = "1.0"

        base_shift_list = parse_string_to_list(base_shift)

        # conditioning
        if isinstance(conditioning, dict) and "encoded" in conditioning:
            cond_texts = conditioning["text"]
            cond_enc   = conditioning["encoded"]
        else:
            cond_texts = [None]
            cond_enc   = [conditioning]

        # подготовка хелперов
        basicguider  = BasicGuider()
        sampler_adv  = SamplerCustomAdvanced()
        latentbatch  = LatentBatch()
        modelsampling = ModelSamplingAuraFlow() if is_flow else ModelSamplingFlux()
        width  = latent_image["samples"].shape[3] * 8
        height = latent_image["samples"].shape[2] * 8

        # LORA
        lora_strength_len = 1
        if loras:
            lora_models    = loras["loras"]
            lora_strengths = loras["strengths"]
            lora_strength_len = sum(len(lst) for lst in lora_strengths)
            self.loraloader = self.loraloader or LoraLoader()

        # прогресс-бар
        total = (
            len(cond_enc) * len(noise_seeds) * len(max_shift_list) * len(base_shift_list)
            * len(guidance_list) * len(sampler_list) * len(scheduler_list)
            * len(steps_list) * len(denoise_list) * lora_strength_len
        )
        pbar = ProgressBar(total) if total > 1 else None

        # выходы
        out_latent = None
        out_params = []
        out_sigmas = None
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

            for c_idx, cond in enumerate(cond_enc):
                prompt = cond_texts[c_idx] if cond_texts[0] else None

                for seed_val in noise_seeds:
                    noise_node = Noise_RandomNoise(seed_val)

                    for ms in max_shift_list:
                        for bs in base_shift_list:
                            if is_flow:
                                work_model = modelsampling.patch_aura(patched_model, bs)[0]
                            else:
                                work_model = modelsampling.patch(patched_model, ms, bs, width, height)[0]

                            for g in guidance_list:
                                cond_val = conditioning_set_values(cond, {"guidance": g})
                                guider = basicguider.get_guider(work_model, cond_val)[0]

                                for samp in sampler_list:
                                    samp_obj = comfy.samplers.sampler_object(samp)

                                    for sched in scheduler_list:
                                        for st in steps_list:
                                            for dn in denoise_list:
                                                sigmas = self._get_sigmas(work_model, sched, st, dn)
                                                out_sigmas = sigmas

                                                counter += 1
                                                logging.info(
                                                    "Sample %d/%d | seed=%s sampler=%s "
                                                    "scheduler=%s steps=%s guidance=%s "
                                                    "max_shift=%s base_shift=%s denoise=%s%s",
                                                    counter, total, seed_val, samp, sched,
                                                    st, g, ms, bs, dn,
                                                    f" lora={lora_models[0]} strength={lora_strengths[0][l_idx]}"
                                                    if loras else ""
                                                )

                                                t0 = time.time()
                                                latent = sampler_adv.sample(
                                                    noise_node, guider, samp_obj, sigmas, latent_image
                                                )[1]
                                                elapsed = time.time() - t0

                                                out_params.append({
                                                    "time":      elapsed,
                                                    "seed":      seed_val,
                                                    "width":     width,
                                                    "height":    height,
                                                    "sampler":   samp,
                                                    "scheduler": sched,
                                                    "steps":     st,
                                                    "guidance":  g,
                                                    "max_shift": ms,
                                                    "base_shift":bs,
                                                    "denoise":   dn,
                                                    "prompt":    prompt,
                                                    **({"lora": lora_models[0],
                                                        "lora_strength": lora_strengths[0][l_idx]} if loras else {})
                                                })

                                                out_latent = (
                                                    latent if out_latent is None
                                                    else latentbatch.batch(out_latent, latent)[0]
                                                )

                                                if pbar:
                                                    pbar.update(1)

        return out_latent, out_params, out_sigmas


# Register node classes
NODE_CLASS_MAPPINGS = {
    "SdxlSamplerParams": SdxlSamplerParams,
}

# Display names for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SdxlSamplerParams": "SdxlSamplerParams (Gayrat)"
}
