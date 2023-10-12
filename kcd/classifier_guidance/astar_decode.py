"""

"""
import torch
from transformers import top_k_top_p_filtering
from tqdm.auto import tqdm

from .fudge_decode import KnowledgeDiscriminator
from .utils import zero_out_after_eos


def astar_generation(
    model=None,
    max_new_tokens: int = 32,
    k: int = 200,
    future_steps: int = 5,
    lambda_weight: float = 0.25,
    disable_adapter_lm_forward=False,
    soft_forward=False,
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
        for i in tqdm(range(max_new_tokens), total=max_new_tokens):
            next_logit, past_key_values = astar_step(
                model,
                discriminator,
                generated_tokens,
                gen_inst_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                top_k=k,
                future_steps=future_steps,
                lambda_weight=lambda_weight,
                disable_adapter_lm_forward=disable_adapter_lm_forward,
                soft_forward=soft_forward)
            generated_tokens = torch.cat([generated_tokens, next_logit], dim=1)
            # early stopping based on eos
            if (generated_tokens == tokenizer.eos_token_id).any(dim=1).all():
                break
        return zero_out_after_eos(generated_tokens, tokenizer.eos_token_id)

    return _generate_fn


@torch.inference_mode()
def astar_step(model,
               discriminator,
               decoded_ids: torch.LongTensor,
               gen_inst_ids: torch.LongTensor = None,
               attention_mask: torch.LongTensor = None,
               encoder_outputs=None,
               past_key_values=None,
               top_k: int = 200,
               top_p: float = 1.0,
               temperature: float = 1.0,
               future_steps: int = 5,
               lambda_weight: float = 0.25,
               disable_adapter_lm_forward: bool = False,
               soft_forward: bool = False):
    assert model.config.is_encoder_decoder
    # prepare input_ids for encoder_decoder or decoder-only models
    if model.config.is_encoder_decoder:
        if gen_inst_ids is None:
            raise ValueError("gen_inst_ids must be set.")
        input_ids = decoded_ids
    else:
        input_ids = decoded_ids
        if gen_inst_ids is not None:
            input_ids = torch.cat([gen_inst_ids, input_ids], dim=1)

    model_inputs = prepare_inputs_for_generation(
        model,
        encoder_outputs=encoder_outputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # first forward: get t+1 topk logits
    outputs = model(**model_inputs,
                    output_attentions=False,
                    output_hidden_states=False)
    if isinstance(outputs, tuple) and len(outputs) == 2:
        outputs, _ = outputs  # ignore token classifier output
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    next_logit = torch.softmax(logits[:, -1, :], dim=-1)  # [B, V]
    values, indices = next_logit.topk(top_k, dim=-1)  # [B, k]
    topk_logits = indices.T  # [k, B]

    # future "rollout" for each topk token
    next_key_values = past_key_values
    all_probs = []
    for logit in topk_logits:
        future_token = logit.unsqueeze(-1)
        future_tokens = [future_token]
        for i in range(future_steps):
            if soft_forward:
                raise NotImplementedError
            # do forward
            else:
                if disable_adapter_lm_forward:
                    with model.disable_adapter():
                        outputs, _ = model(encoder_outputs=encoder_outputs,
                                           attention_mask=attention_mask,
                                           past_key_values=next_key_values,
                                           decoder_input_ids=future_token,
                                           use_cache=True)
                else:
                    outputs, _ = model(encoder_outputs=encoder_outputs,
                                       attention_mask=attention_mask,
                                       past_key_values=next_key_values,
                                       decoder_input_ids=future_token,
                                       use_cache=True)
                future_logits = outputs.logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(future_logits,
                                                        top_p=top_p,
                                                        top_k=top_k)
                probs = torch.softmax(filtered_logits, dim=-1)
                future_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                next_key_values = outputs.past_key_values
                future_tokens.append(future_token)
        future_tokens = torch.cat(future_tokens, dim=1)  # [B, future_steps + 1]
        class_prob = discriminator(torch.cat([decoded_ids, future_tokens], dim=1),
                                   future_token.T,  # [1, B]
                                   gen_inst_ids=gen_inst_ids,
                                   attention_mask=attention_mask,
                                   encoder_outputs=encoder_outputs)  # [B,]
        all_probs.append(class_prob)
    all_probs = torch.stack(all_probs, dim=1)  # [B, k]
    astar_topk = values + lambda_weight * all_probs  # [B, k]
    max_idx = astar_topk.argmax(dim=-1)  # [B,]
    max_logit = indices[range(indices.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]

    return max_logit, past_key_values


def prepare_inputs_for_generation(model, encoder_outputs=None, **kwargs):
    if model.config.is_encoder_decoder:
        model_inputs = model.prepare_inputs_for_generation(encoder_outputs=encoder_outputs,
                                                           **kwargs)
    else:
        model_inputs = model.prepare_inputs_for_generation(**kwargs)
    return model_inputs
