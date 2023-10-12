"""

"""
import torch
from tqdm.auto import tqdm
from transformers import top_k_top_p_filtering

from .fudge_decode import KnowledgeDiscriminator
from .utils import zero_out_after_eos

def nado_generation(
    model=None,
    max_new_tokens: int = 32,
    k: int = 200,
    alpha: int = 1.0,
    disable_adapter_lm_forward=False,
):
    assert model is not None
    discriminator = KnowledgeDiscriminator(model)

    def _generate_fn(model, tokenizer, inputs):
        """input_ids: torch.LongTensor, knowledge_text: str"""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if model.config.is_encoder_decoder:
            encoder_outputs = model.get_encoder()(input_ids=input_ids,
                                                  attention_mask=attention_mask)
        else:
            encoder_outputs = None
        past_key_values = None
        generated_tokens = torch.full((input_ids.shape[0], 1),
                                      model.config.decoder_start_token_id,
                                      dtype=torch.long,
                                      device=input_ids.device)
        score = torch.ones_like(generated_tokens, dtype=torch.float)
        for i in tqdm(range(max_new_tokens), total=max_new_tokens):
            next_logit, score, past_key_values = nado_step(
                model,
                discriminator,
                generated_tokens,
                score,
                gen_inst_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                k=k,
                alpha=alpha,
                disable_adapter_lm_forward=disable_adapter_lm_forward)
            generated_tokens = torch.cat([generated_tokens, next_logit], dim=1)
            # early stopping based on eos
            if (generated_tokens == tokenizer.eos_token_id).any(dim=1).all():
                break
        return zero_out_after_eos(generated_tokens, tokenizer.eos_token_id)

    return _generate_fn


@torch.inference_mode()
def nado_step(model,
              discriminator,
              decoded_ids: torch.LongTensor,
              current_ids_score: torch.Tensor,
              gen_inst_ids: torch.LongTensor = None,
              attention_mask: torch.LongTensor = None,
              encoder_outputs=None,
              past_key_values=None,
              k: int = 200,
              alpha: float = 1.0,
              disable_adapter_lm_forward: bool = False):
    assert model.config.is_encoder_decoder
    # check v2 discriminator
    if hasattr(model, 'base_model'):  # peft
        v2 = getattr(model.base_model, 'v2', False)
    else:
        v2 = getattr(model, 'v2', False)

    # prepare input_ids for encoder_decoder or decoder-only models
    if model.config.is_encoder_decoder:
        if gen_inst_ids is None:
            raise ValueError("gen_inst_ids must be set.")
        input_ids = decoded_ids
    else:
        input_ids = decoded_ids
        if gen_inst_ids is not None:
            input_ids = torch.cat([gen_inst_ids, input_ids], dim=1)

    if model.config.is_encoder_decoder:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True)
    else:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True)

    # do forward
    if disable_adapter_lm_forward:
        with model.disable_adapter():
            outputs = model(**model_inputs,
                            return_lm_only=True,
                            output_attentions=False,
                            output_hidden_states=False)
        if v2:
            # with adapter
            _, disc_outputs = model(**model_inputs,
                                    output_attentions=False,
                                    output_hidden_states=False)
            class_logit = disc_outputs.logits
    else:
        outputs = model(**model_inputs,
                        output_attentions=False,
                        output_hidden_states=False)
        if v2:
            outputs, disc_out = outputs
            class_logit = disc_out.logits  # [B, V]

    # select the next logit
    if isinstance(outputs, tuple) and len(outputs) == 2:
        outputs, _ = outputs  # ignore token classifier output

    next_key_values = outputs.past_key_values
    logits = outputs.logits

    if v2:
        class_logp = torch.nn.LogSigmoid()(class_logit)  # [B, V]
        logits = logits[:, -1]
        logits = top_k_top_p_filtering(logits, top_k=k)
        next_logit = torch.log_softmax(logits, dim=-1)
        next_logit = next_logit + class_logp * alpha
        _, max_logit = next_logit.topk(1, dim=-1)  # [B, 1]
        score = class_logp[range(class_logp.shape[0]), max_logit.squeeze(-1)]
        return max_logit, score, next_key_values


    next_logit = torch.softmax(logits[:, -1, :], dim=-1)  # [B, V]
    values, indices = next_logit.topk(k, dim=-1)  # [B, k]
    topk_logits = indices.T  # [k, B]

    class_prob = discriminator(decoded_ids,
                               topk_logits,
                               gen_inst_ids=gen_inst_ids,
                               attention_mask=attention_mask,
                               encoder_outputs=encoder_outputs)  # [K * B,]
    class_prob = class_prob.view(k, -1).T  # [K, B] => [B, K]
    class_prob = class_prob / current_ids_score

    fudge_topk = values * class_prob  # [B, K,]
    max_idx = fudge_topk.argmax(dim=-1)  # [B,]
    max_logit = indices[range(indices.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]
    score = class_prob[range(indices.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]
    return max_logit, score, next_key_values
