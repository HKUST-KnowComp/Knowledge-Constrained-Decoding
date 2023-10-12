from .guided_generation_predictor import GuidedGenerationPredictor
from .metric_guidance import metric_guided_generation
from .fudge_decode import fudge_generation
from .openai_fudge_decode import openai_fudge_generation
from .nado_decode import nado_generation
from .astar_decode import astar_generation

GENERATE_FN_REGISTRY = {
    'metric_guidance': metric_guided_generation,
    'fudge': fudge_generation,
    'nado': nado_generation,
    'astar': astar_generation,
    'openai_fudge': openai_fudge_generation
}


def load_generate_fn(name, **kwargs):
    return GENERATE_FN_REGISTRY[name](**kwargs)
