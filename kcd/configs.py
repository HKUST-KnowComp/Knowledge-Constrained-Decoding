from dataclasses import dataclass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 32
    do_sample: bool = True
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 0.95
    num_return_sequences: int = 1
    top_k: int = 200
