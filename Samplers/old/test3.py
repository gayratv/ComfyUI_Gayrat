import comfy.samplers
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
# from node_helpers import conditioning_set_values, parse_string_to_list
from nodes import LoraLoader
from comfy.utils import ProgressBar

from nodes import BasicGuider

from nodes import (
    # BasicGuider,
    SamplerCustomAdvanced,
    LatentBatch,
    ModelSamplingFlux,
    ModelSamplingAuraFlow,
    Noise_RandomNoise,
    ProgressBar,
    OptimalStepsScheduler,       # ← наш «особый» узел
    LoraLoader,
)