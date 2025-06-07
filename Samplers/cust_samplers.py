import json
import random
import time
import logging
import torch
import comfy.samplers
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
from node_helpers import conditioning_set_values, parse_string_to_list
from nodes import LoraLoader
from comfy.utils import ProgressBar

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

# Register node classes
NODE_CLASS_MAPPINGS = {
    "SamplerSelectHelper": SamplerSelectHelper,
    "SchedulerSelectHelper": SchedulerSelectHelper,
    "FluxSamplerParams": FluxSamplerParams,
}

# Display names for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSelectHelper": "Sampler Select Helper",
    "SchedulerSelectHelper": "Scheduler Select Helper",
    "FluxSamplerParams": "Flux Sampler Params",
}
