from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import nn
from transformers import Seq2SeqTrainer
from transformers.deepspeed import is_deepspeed_zero3_enabled


class GuidedGenerationPredictor(Seq2SeqTrainer):

    def __init__(self, generate_fn: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.generate_fn = generate_fn

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: List[str] | None = None
    ) -> Tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Copied from Seq2SeqTrainer.prediction_step, but with the following changes:
        - use `metric_guidance` instaed of model.generate
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model,
                                           inputs,
                                           prediction_loss_only=prediction_loss_only,
                                           ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (gen_kwargs["num_beams"] if gen_kwargs.get("num_beams")
                                   is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus")
                                     is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ("labels" in inputs and "decoder_input_ids" in inputs and
                inputs["labels"].shape == inputs["decoder_input_ids"].shape):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        ############## NOTE: changes starts here ###############################
        generated_tokens = self.generate_fn(model, self.tokenizer, inputs)
        return None, generated_tokens, None
