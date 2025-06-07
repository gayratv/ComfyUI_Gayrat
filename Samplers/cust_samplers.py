import json
import random
import time
import logging
import torch
import os

import comfy.samplers
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
# from node_helpers import conditioning_set_values, parse_string_to_list
from nodes import LoraLoader
from comfy.utils import ProgressBar

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
        selected = [name for name, val in values.items() if val]
        return (", ".join(selected),)

# Flux sampler parameters with internal scheduler helper
class FluxSamplerParams:
    class _BasicScheduler:
        """
        Internal helper for computing sigma schedules.
        Not registered as a separate node.
        """
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }}
        RETURN_TYPES = ("SIGMAS",)
        FUNCTION = "get_sigmas"

        def get_sigmas(self, model, scheduler, steps, denoise):
            total_steps = steps
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                total_steps = int(steps / denoise)
            sigmas = comfy.samplers.calculate_sigmas(
                model.get_model_object("model_sampling"), scheduler, total_steps
            ).cpu()
            sigmas = sigmas[-(steps + 1):]
            return (sigmas,)

    def __init__(self):
        self.loraloader = None
        self._scheduler = self._BasicScheduler()
        self.lora = (None, None)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "?"}),
            "sampler": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "euler"}),
            "scheduler": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "simple"}),
            "steps": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "20"}),
            "guidance": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "3.5"}),
            "max_shift": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": ""}),
            "base_shift": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": ""}),
            "denoise": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "1.0"}),
        }, "optional": {
            "loras": ("LORA_PARAMS",),
        }}

    RETURN_TYPES = ("LATENT", "SAMPLER_PARAMS", "SIGMAS")
    RETURN_NAMES = ("latent", "params", "sigmas")
    FUNCTION = "execute"
    CATEGORY = "Gayrat/sampling"

    def execute(self, model, conditioning, latent_image, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise, loras=None):
        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        # Parse seeds
        noise = seed.replace("\n", ",").split(",")
        noise = [random.randint(0, 999999) if "?" in n else int(n) for n in noise]
        if not noise:
            noise = [random.randint(0, 999999)]

        # Sampler list
        if sampler == '*':
            sampler_list = comfy.samplers.KSampler.SAMPLERS
        elif sampler.startswith("!"):
            exclude = [s.strip("! ") for s in sampler.replace("\n", ",").split(",")]
            sampler_list = [s for s in comfy.samplers.KSampler.SAMPLERS if s not in exclude]
        else:
            sampler_list = [s.strip() for s in sampler.replace("\n", ",").split(",") if s.strip() in comfy.samplers.KSampler.SAMPLERS]
        if not sampler_list:
            sampler_list = ['ipndm']

        # Scheduler list
        if scheduler == '*':
            scheduler_list = comfy.samplers.KSampler.SCHEDULERS
        elif scheduler.startswith("!"):
            exclude = [s.strip("! ") for s in scheduler.replace("\n", ",").split(",")]
            scheduler_list = [s for s in comfy.samplers.KSampler.SCHEDULERS if s not in exclude]
        else:
            scheduler_list = [s.strip() for s in scheduler.replace("\n", ",").split(",") if s in comfy.samplers.KSampler.SCHEDULERS]
        if not scheduler_list:
            scheduler_list = ['simple']

        # Parse numeric lists
        steps_list = parse_string_to_list(steps or ("4" if is_schnell else "20"))
        denoise_list = parse_string_to_list(denoise or "1.0")
        guidance_list = parse_string_to_list(guidance or "3.5")
        max_shift_list = parse_string_to_list(max_shift or ("0" if is_schnell else "1.15"))
        base_shift_list = parse_string_to_list(base_shift or ("1.0" if is_schnell else "0.5"))

        # Conditioning
        cond_text = None
        if isinstance(conditioning, dict) and "encoded" in conditioning:
            cond_text = conditioning["text"]
            cond_encoded = conditioning["encoded"]
        else:
            cond_encoded = [conditioning]

        out_latent = None
        out_params = []
        out_sigmas = None

        # Initialize helper nodes
        basicguider = BasicGuider()
        sampler_adv = SamplerCustomAdvanced()
        latentbatch = LatentBatch()
        modelsampling = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()
        width = latent_image["samples"].shape[3] * 8
        height = latent_image["samples"].shape[2] * 8

        # LORA setup
        lora_strength_len = 1
        if loras:
            lora_model = loras["loras"]
            lora_strengths = loras["strengths"]
            lora_strength_len = sum(len(lst) for lst in lora_strengths)
            if self.loraloader is None:
                self.loraloader = LoraLoader()

        total_samples = (
            len(cond_encoded) * len(noise) * len(max_shift_list) * len(base_shift_list)
            * len(guidance_list) * len(sampler_list) * len(scheduler_list)
            * len(steps_list) * len(denoise_list) * lora_strength_len
        )
        current = 0
        pbar = ProgressBar(total_samples) if total_samples > 1 else None

        for los in range(lora_strength_len):
            patched_model = (
                self.loraloader.load_lora(model, None, lora_model[0], lora_strengths[0][los], 0)[0]
                if loras else model
            )
            for idx, cond in enumerate(cond_encoded):
                ct = cond_text[idx] if cond_text else None
                for n in noise:
                    noise_node = Noise_RandomNoise(n)
                    for ms in max_shift_list:
                        for bs in base_shift_list:
                            work_model = (
                                modelsampling.patch_aura(patched_model, bs)[0]
                                if is_schnell else modelsampling.patch(patched_model, ms, bs, width, height)[0]
                            )
                            for g in guidance_list:
                                cond_val = conditioning_set_values(cond, {"guidance": g})
                                guider = basicguider.get_guider(work_model, cond_val)[0]
                                for s in sampler_list:
                                    sampler_obj = comfy.samplers.sampler_object(s)
                                    for sc in scheduler_list:
                                        for st in steps_list:
                                            for d in denoise_list:
                                                # Compute sigma schedule via internal helper
                                                sigmas = self._scheduler.get_sigmas(work_model, sc, st, d)[0]
                                                out_sigmas = sigmas
                                                current += 1
                                                log = (
                                                    f"Sampling {current}/{total_samples} with seed {n}, sampler {s}, "
                                                    f"scheduler {sc}, steps {st}, guidance {g}, max_shift {ms}, "
                                                    f"base_shift {bs}, denoise {d}"
                                                )
                                                if loras:
                                                    log += f", lora {lora_model[0]}, strength {lora_strengths[0][los]}"
                                                logging.info(log)
                                                start = time.time()
                                                latent = sampler_adv.sample(
                                                    noise_node, guider, sampler_obj, sigmas, latent_image
                                                )[1]
                                                elapsed = time.time() - start
                                                out_params.append({
                                                    "time": elapsed,
                                                    "seed": n,
                                                    "width": width,
                                                    "height": height,
                                                    "sampler": s,
                                                    "scheduler": sc,
                                                    "steps": st,
                                                    "guidance": g,
                                                    "max_shift": ms,
                                                    "base_shift": bs,
                                                    "denoise": d,
                                                    "prompt": ct,
                                                    **({"lora": lora_model[0], "lora_strength": lora_strengths[0][los]} if loras else {})
                                                })
                                                out_latent = (
                                                    latent if out_latent is None else latentbatch.batch(out_latent, latent)[0]
                                                )
                                                if pbar:
                                                    pbar.update(1)

        return (out_latent, out_params, out_sigmas)

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
