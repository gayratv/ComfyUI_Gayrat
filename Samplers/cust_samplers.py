# ──────────────────────────────────────────────────────────────────────────────
#  FluxSamplerParams  – модифицированная версия
#  • Поддерживает кастомные планировщики (например, OptimalStepsScheduler)
#  • Для имени planировщика "optimal"/"optimal_steps"/"optimalsteps" вызывает
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

# ──   узлы/утилиты, которые уже есть в ComfyUI  ───────────────────────────────
from nodes import (
    BasicGuider,
    SamplerCustomAdvanced,
    LatentBatch,
    ModelSamplingFlux,
    ModelSamplingAuraFlow,
    Noise_RandomNoise,
    ProgressBar,
    OptimalStepsScheduler,       # ← наш «особый» узел
    LoraLoader,
)

from utils import (
    parse_string_to_list,
    conditioning_set_values,
)


import torch.nn.functional as F
# import torchvision.transforms.v2 as T
import torchvision
T = torchvision.transforms.v2


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


# Sampler selection helper
class SamplerSelectHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            **{s: ("BOOLEAN", {"default": False}) for s in comfy.samplers.KSampler.SAMPLERS},
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Gayrat/sampling"

    def execute(self, **values):
        selected = [name for name, val in values.items() if val]
        return (", ".join(selected),)

# Scheduler selection helper
class SchedulerSelectHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            **{s: ("BOOLEAN", {"default": False}) for s in comfy.samplers.KSampler.SCHEDULERS},
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Gayrat/sampling"

    def execute(self, **values):
        """
        Returns a comma-separated list of selected schedulers.
        Ensures 'OptimalStepsScheduler' is always present in the output.
        """
        selected = [name for name, val in values.items() if val]
        # Always include OptimalStepsScheduler
        if "OptimalStepsScheduler" not in selected:
            selected.append("OptimalStepsScheduler")
        return (", ".join(selected),)

# Flux sampler parameters with internal scheduler helper
# ──────────────────────────────────────────────────────────────────────────────
class FluxSamplerParams:
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
        if sched_l in {"optimal", "optimal_steps", "optimalsteps"}:
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
                "seed": ("STRING", {"default": "?"}),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "steps": ("STRING", {"default": "20"}),
                "guidance": ("STRING", {"default": "3.5"}),
                "max_shift": ("STRING", {"default": ""}),
                "base_shift": ("STRING", {"default": ""}),
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
        is_flow = model.model.model_type == comfy.model_base.ModelType.FLOW

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
        base_shift_list= parse_string_to_list(base_shift  ("1.0" if is_flow else "0.5"))

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
                            work_model = (
                                modelsampling.patch_aura(patched_model, bs)[0]
                                if is_flow else
                                modelsampling.patch(patched_model, ms, bs, width, height)[0]
                            )

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

class PlotParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "params": ("SAMPLER_PARAMS", ),
                    "order_by": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler", "guidance", "max_shift", "base_shift", "lora_strength"], ),
                    "cols_value": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler", "guidance", "max_shift", "base_shift", "lora_strength"], ),
                    "cols_num": ("INT", {"default": -1, "min": -1, "max": 1024 }),
                    "add_prompt": (["false", "true", "excerpt"], ),
                    "add_params": (["false", "true", "changes only"], {"default": "true"}),
                }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "Gayrat/sampling"

    def execute(self, images, params, order_by, cols_value, cols_num, add_prompt, add_params):
        from PIL import Image, ImageDraw, ImageFont
        import math
        import textwrap

        if images.shape[0] != len(params):
            raise ValueError("Number of images and number of parameters do not match.")

        _params = params.copy()

        if order_by != "none":
            sorted_params = sorted(_params, key=lambda x: x[order_by])
            indices = [_params.index(item) for item in sorted_params]
            images = images[torch.tensor(indices)]
            _params = sorted_params

        if cols_value != "none" and cols_num > -1:
            groups = {}
            for p in _params:
                value = p[cols_value]
                if value not in groups:
                    groups[value] = []
                groups[value].append(p)
            cols_num = len(groups)

            sorted_params = []
            groups = list(groups.values())
            for g in zip(*groups):
                sorted_params.extend(g)

            indices = [_params.index(item) for item in sorted_params]
            images = images[torch.tensor(indices)]
            _params = sorted_params
        elif cols_num == 0:
            cols_num = int(math.sqrt(images.shape[0]))
            cols_num = max(1, min(cols_num, 1024))

        width = images.shape[2]
        out_image = []

        font = ImageFont.truetype(os.path.join(FONTS_DIR, 'ShareTechMono-Regular.ttf'), min(48, int(32*(width/1024))))
        text_padding = 3
        line_height = font.getmask('Q').getbbox()[3] + font.getmetrics()[1] + text_padding*2
        char_width = font.getbbox('M')[2]+1 # using monospace font

        if add_params == "changes only":
            value_tracker = {}
            for p in _params:
                for key, value in p.items():
                    if key != "time":
                        if key not in value_tracker:
                            value_tracker[key] = set()
                        value_tracker[key].add(value)
            changing_keys = {key for key, values in value_tracker.items() if len(values) > 1 or key == "prompt"}

            result = []
            for p in _params:
                changing_params = {key: value for key, value in p.items() if key in changing_keys}
                result.append(changing_params)

            _params = result

        for (image, param) in zip(images, _params):
            image = image.permute(2, 0, 1)

            if add_params != "false":
                if add_params == "changes only":
                    text = "\n".join([f"{key}: {value}" for key, value in param.items() if key != "prompt"])
                else:
                    text = f"time: {param['time']:.2f}s, seed: {param['seed']}, steps: {param['steps']}, size: {param['width']}×{param['height']}\ndenoise: {param['denoise']}, sampler: {param['sampler']}, sched: {param['scheduler']}\nguidance: {param['guidance']}, max/base shift: {param['max_shift']}/{param['base_shift']}"
                    if 'lora' in param and param['lora']:
                        text += f"\nLoRA: {param['lora'][:32]}, str: {param['lora_strength']}"

                lines = text.split("\n")
                text_height = line_height * len(lines)
                text_image = Image.new('RGB', (width, text_height), color=(0, 0, 0))

                for i, line in enumerate(lines):
                    draw = ImageDraw.Draw(text_image)
                    draw.text((text_padding, i * line_height + text_padding), line, font=font, fill=(255, 255, 255))

                # text_image = T.ToTensor()(text_image).to(image.device)
                text_image = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(text_image).to(image.device)
                image = torch.cat([image, text_image], 1)

            if 'prompt' in param and param['prompt'] and add_prompt != "false":
                prompt = param['prompt']
                if add_prompt == "excerpt":
                    prompt = " ".join(param['prompt'].split()[:64])
                    prompt += "..."

                cols = math.ceil(width / char_width)
                prompt_lines = textwrap.wrap(prompt, width=cols)
                prompt_height = line_height * len(prompt_lines)
                prompt_image = Image.new('RGB', (width, prompt_height), color=(0, 0, 0))

                for i, line in enumerate(prompt_lines):
                    draw = ImageDraw.Draw(prompt_image)
                    draw.text((text_padding, i * line_height + text_padding), line, font=font, fill=(255, 255, 255))

                # prompt_image = T.ToTensor()(prompt_image).to(image.device)
                prompt_image = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(prompt_image).to(image.device)
                image = torch.cat([image, prompt_image], 1)

            # a little cleanup
            image = torch.nan_to_num(image, nan=0.0).clamp(0.0, 1.0)
            out_image.append(image)

        # ensure all images have the same height
        if add_prompt != "false" or add_params == "changes only":
            max_height = max([image.shape[1] for image in out_image])
            out_image = [F.pad(image, (0, 0, 0, max_height - image.shape[1])) for image in out_image]

        out_image = torch.stack(out_image, 0).permute(0, 2, 3, 1)

        # merge images
        if cols_num > -1:
            cols = min(cols_num, out_image.shape[0])
            b, h, w, c = out_image.shape
            rows = math.ceil(b / cols)

            # Pad the tensor if necessary
            if b % cols != 0:
                padding = cols - (b % cols)
                out_image = F.pad(out_image, (0, 0, 0, 0, 0, 0, 0, padding))
                b = out_image.shape[0]

            # Reshape and transpose
            out_image = out_image.reshape(rows, cols, h, w, c)
            out_image = out_image.permute(0, 2, 1, 3, 4)
            out_image = out_image.reshape(rows * h, cols * w, c).unsqueeze(0)

        return (out_image, )

# Register node classes
NODE_CLASS_MAPPINGS = {
    "SamplerSelectHelper": SamplerSelectHelper,
    "SchedulerSelectHelper": SchedulerSelectHelper,
    "FluxSamplerParams": FluxSamplerParams,
    "PlotParameters+": PlotParameters
}

# Display names for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSelectHelper": "Sampler Select Helper",
    "SchedulerSelectHelper": "Scheduler Select Helper",
    "FluxSamplerParams": "Flux Sampler Params",
    "PlotParameters+": "🔧 Plot Sampler Parameters"
}
