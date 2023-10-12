"""

"""
import math

import torch
from tqdm.auto import tqdm
from transformers import top_k_top_p_filtering

from .utils import zero_out_after_eos


class ZeroShotDiscriminator:

    def __init__(self, model, tokenizer, pos_attr_idx, neg_attr_idx,
                 classification_instruction) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pos_attr_idx = pos_attr_idx
        self.neg_attr_idx = neg_attr_idx
        self.instruction = classification_instruction

    def __call__(self, candidates: list[torch.LongTensor]):
        decoder_start_token_id = torch.LongTensor([[self.model.config.decoder_start_token_id]])
        decoder_start_token_id = decoder_start_token_id.to(candidates.device)

        all_text = []
        for cand in candidates:
            cand_str = self.tokenizer.batch_decode(cand)[0]  # NOTE: assume bsz=1
            classification_prompt = self.instruction.format(cand_str)
            all_text.append(classification_prompt)

        inputs = self.tokenizer(all_text, return_tensors='pt', padding=True).to(candidates.device)
        out = self.model(**inputs, decoder_input_ids=decoder_start_token_id)
        class_prob = torch.softmax(out.logits[:, 0, [self.pos_attr_idx, self.neg_attr_idx]],
                                   dim=-1)  # [K, 2]
        class_prob = class_prob[:, 0]  # postive
        return class_prob


class KnowledgeDiscriminator:

    def __init__(self, model, pos_attr_idx=1) -> None:
        self.model = model
        self.pos_attr_idx = pos_attr_idx

    def __call__(self,
                 decoded_ids: torch.LongTensor,
                 topk_logits: torch.LongTensor,
                 gen_inst_ids: torch.LongTensor = None,
                 attention_mask: torch.LongTensor = None,
                 encoder_outputs=None):
        if encoder_outputs is None:
            encoder_outputs = self.model.get_encoder()(input_ids=gen_inst_ids,
                                                       attention_mask=attention_mask)

        past_key_values = self.model(encoder_outputs=encoder_outputs,
                                     attention_mask=attention_mask,
                                     decoder_input_ids=decoded_ids,
                                     use_cache=True)[0].past_key_values
        all_probs = []
        for logit in topk_logits:
            _, token_class_outs = self.model(encoder_outputs=encoder_outputs,
                                             attention_mask=attention_mask,
                                             past_key_values=past_key_values,
                                             decoder_input_ids=logit.unsqueeze(-1),
                                             use_cache=True)
            probs = torch.softmax(token_class_outs.logits, dim=-1)  # [B, 2]
            all_probs.append(probs)
        probs = torch.cat(all_probs, dim=0)  # [N, 2]
        return probs[:, self.pos_attr_idx]


def fudge_generation(
    model=None,
    max_new_tokens: int = 32,
    k: int = 200,
    disable_adapter_lm_forward=False,
    complete_after: int = 0,
):
    assert model is not None
    discriminator = KnowledgeDiscriminator(model)

    def _generate_fn(model, tokenizer, inputs):
        """input_ids: torch.LongTensor, knowledge_text: str"""
        input_ids = inputs['input_ids']
        # knowledge_text = tokenizer.batch_decode(inputs['knowledge_ids'])
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
        if complete_after > 0:
            tokens_to_gen = complete_after
        else:
            tokens_to_gen = max_new_tokens
        early_stop = False
        for i in tqdm(range(tokens_to_gen), total=tokens_to_gen):
            next_logit, past_key_values = fudge_step(
                model,
                discriminator,
                generated_tokens,
                gen_inst_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                k=k,
                disable_adapter_lm_forward=disable_adapter_lm_forward)
            generated_tokens = torch.cat([generated_tokens, next_logit], dim=1)
            # early stopping based on eos
            if (generated_tokens == tokenizer.eos_token_id).any(dim=1).all():
                early_stop = True
                break
        if not early_stop and complete_after > 0:
            tokens_to_gen = max_new_tokens - complete_after
            if disable_adapter_lm_forward:
                model.base_model.base_model.return_lm_only = True
                with model.disable_adapter():
                    generated_tokens = model.generate(input_ids=input_ids,
                                                      encoder_outputs=encoder_outputs,
                                                      attention_mask=attention_mask,
                                                      decoder_input_ids=generated_tokens,
                                                      max_new_tokens=tokens_to_gen)
                model.base_model.base_model.return_lm_only = False
            else:
                model.return_lm_only = True
                generated_tokens = model.generate(input_ids=input_ids,
                                                  encoder_outputs=encoder_outputs,
                                                  attention_mask=attention_mask,
                                                  decoder_input_ids=generated_tokens,
                                                  max_new_tokens=tokens_to_gen)
                model.return_lm_only = False

        return zero_out_after_eos(generated_tokens, tokenizer.eos_token_id)

    return _generate_fn


@torch.inference_mode()
def fudge_step(model,
               discriminator,
               decoded_ids: torch.LongTensor,
               gen_inst_ids: torch.LongTensor = None,
               attention_mask: torch.LongTensor = None,
               encoder_outputs=None,
               past_key_values=None,
               k: int = 200,
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
        model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids,
                                                           encoder_outputs=encoder_outputs,
                                                           attention_mask=attention_mask,
                                                           past_key_values=past_key_values,
                                                           use_cache=True)
    else:
        model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids,
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
        outputs = model(**model_inputs, output_attentions=False, output_hidden_states=False)
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
        next_logit = next_logit + class_logp
        _, max_logit = next_logit.topk(1, dim=-1)  # [B, 1]
        return max_logit, next_key_values

    next_logit = torch.softmax(logits[:, -1, :], dim=-1)  # [B, V]
    values, indices = next_logit.topk(k, dim=-1)  # [B, k]
    topk_logits = indices.T  # [k, B]

    class_prob = discriminator(decoded_ids,
                               topk_logits,
                               gen_inst_ids=gen_inst_ids,
                               attention_mask=attention_mask,
                               encoder_outputs=encoder_outputs)  # [K * B,]
    class_prob = class_prob.view(k, -1).T  # [K, B] => [B, K]

    fudge_topk = values * class_prob  # [B, K,]
    max_idx = fudge_topk.argmax(dim=-1)  # [B,]
    max_logit = indices[range(indices.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]
    return max_logit, next_key_values
