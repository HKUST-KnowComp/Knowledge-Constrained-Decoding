"""

"""
import time
import traceback

import torch
from tqdm.auto import tqdm
from transformers import top_k_top_p_filtering
import tiktoken
# from .utils import zero_out_after_eos


class KnowledgeDiscriminator:

    def __init__(self,
                 model,
                 tokenizer,
                 chatgpt_tokenizer=None,
                 pos_attr_idx=1,
                 bias_scaling_factor=2.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chatgpt_tokenizer = chatgpt_tokenizer
        self.pos_attr_idx = pos_attr_idx
        self.is_peft = hasattr(model, 'base_model')
        self.bias_scaling_factor = bias_scaling_factor

    def __call__(self,
                 decoded_ids: torch.LongTensor,
                 topk_logits: torch.LongTensor,
                 return_logprobs: bool = True,
                 encoder_input_ids: torch.LongTensor = None,
                 attention_mask: torch.LongTensor = None):
        if self.model.config.is_encoder_decoder:
            past_key_values = self.model(input_ids=encoder_input_ids,
                                         attention_mask=attention_mask,
                                         decoder_input_ids=decoded_ids,
                                         return_lm_only=True,
                                         use_cache=True).past_key_values
        else:
            past_key_values = self.model(input_ids=decoded_ids,
                                         attention_mask=attention_mask,
                                         return_lm_only=True,
                                         use_cache=True).past_key_values
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (decoded_ids.shape[0], 1), dtype=torch.long, device=decoded_ids.device)
                ],
                                           dim=1)
        all_probs = []
        for logit in topk_logits:
            if self.model.config.is_encoder_decoder:
                _, logits = self.model(input_ids=encoder_input_ids,
                                       attention_mask=attention_mask,
                                       decoder_input_ids=logit.unsqueeze(-1),
                                       past_key_values=past_key_values,
                                       use_cache=True)
            else:
                _, logits = self.model(input_ids=logit.unsqueeze(-1),
                                       attention_mask=attention_mask,
                                       past_key_values=past_key_values,
                                       use_cache=True)
            if return_logprobs:
                probs = torch.log_softmax(logits, dim=-1)  # [B, 2]
            else:
                probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
        probs = torch.cat(all_probs, dim=0)  # [N, 2]
        return probs[:, self.pos_attr_idx]

    def propose_reranked_topk(self,
                              decoded_ids: torch.LongTensor,
                              k: int = 50,
                              encoder_input_ids: torch.LongTensor = None,
                              attention_mask: torch.LongTensor = None):
        if self.model.config.is_encoder_decoder:
            kwargs = {
                'input_ids': encoder_input_ids,
                'attention_mask': attention_mask,
                'decoder_input_ids': decoded_ids,
                'return_lm_only': True,
            }
        else:
            kwargs = {
                'input_ids': decoded_ids,
                'attention_mask': attention_mask,
                'return_lm_only': True,
            }
        if self.is_peft:
            with self.model.disable_adapter():
                logits = self.model(**kwargs).logits
        else:
            logits = self.model(**kwargs).logits

        logits = logits[:, -1, :]  # [B, V]
        logits = top_k_top_p_filtering(logits, top_k=k)
        probs = torch.softmax(logits, dim=-1)  # [B, V]
        probs, indices = probs.topk(k=k, dim=-1)  # [B, k]
        class_probs = self(decoded_ids,
                           indices.T,
                           attention_mask=attention_mask,
                           encoder_input_ids=encoder_input_ids,
                           return_logprobs=False)  # [B, k]
        class_probs = class_probs.view(k, -1).T  # [K, B] => [B, K]
        class_probs = torch.where(class_probs > 0.5, class_probs, -1)
        final_prob = probs * class_probs  # [B, k], between 0 ~ 1
        logit_bias = final_prob * self.bias_scaling_factor

        batch_logit_bias_map = []
        for bias, idx in zip(logit_bias, indices):
            logit_bias_map = {}
            for b, i in zip(bias, idx):
                if self.chatgpt_tokenizer is not None:
                    # convert tokens to chatGPT tokenizer indices
                    tokens = self.chatgpt_tokenizer.encode(self.tokenizer.decode(i.item()),
                                                           allowed_special="all")
                    for tok in tokens:
                        logit_bias_map[str(tok)] = b.item()
                else:
                    logit_bias_map[str(i.item())] = b.item()
            batch_logit_bias_map.append(logit_bias_map)
        return batch_logit_bias_map


def openai_fudge_generation(
    openai_model=None,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 32,
    k: int = 6,
    parameters=None,
    use_logit_bias=False,
    pre_post_guidance=False,
    propose_topk=50,
):
    assert openai_model is not None and model is not None and parameters is not None
    assert tokenizer is not None
    chatgpt_tokenizer = None
    if openai_model.model_name == 'gpt-3.5-turbo':  # chatgpt
        chatgpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    discriminator = KnowledgeDiscriminator(model, tokenizer, chatgpt_tokenizer=chatgpt_tokenizer)

    def _generate_fn(inputs):
        """input_ids: torch.LongTensor, knowledge_text: str"""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_usage = 0
        if model.config.is_encoder_decoder:
            bsz = input_ids.shape[0]
            generated_tokens = model.config.decoder_start_token_id * torch.ones(
                (bsz, 1), device=input_ids.device, dtype=torch.long)
        else:
            generated_tokens = input_ids
        for i in tqdm(range(max_new_tokens), total=max_new_tokens):
            try:
                next_logit, usages = openai_fudge_step(openai_model,
                                                       parameters,
                                                       tokenizer,
                                                       discriminator,
                                                       generated_tokens,
                                                       use_logit_bias=use_logit_bias,
                                                       pre_post_guidance=pre_post_guidance,
                                                       propose_topk=propose_topk,
                                                       encoder_input_ids=input_ids,
                                                       attention_mask=attention_mask)
            except Exception as err:
                traceback.print_tb(err.__traceback__)
                # save up to now
                return generated_tokens, token_usage, False
            generated_tokens = torch.cat([generated_tokens, next_logit], dim=1)
            if not model.config.is_encoder_decoder:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
                ],
                                           dim=1)
            token_usage += usages
            # early stopping based on eos
            # if (generated_tokens == tokenizer.eos_token_id).any(dim=1).all():
            #     break
        return generated_tokens, token_usage, True

    return _generate_fn


@torch.inference_mode()
def openai_fudge_step(openai_model,
                      parameters,
                      tokenizer,
                      discriminator,
                      decoded_ids: torch.LongTensor,
                      use_logit_bias: bool = False,
                      pre_post_guidance: bool = False,
                      propose_topk: int = 50,
                      encoder_input_ids: torch.LongTensor = None,
                      attention_mask: torch.LongTensor = None):
    logit_bias = None
    if use_logit_bias:
        logit_bias = discriminator.propose_reranked_topk(decoded_ids,
                                                         k=propose_topk,
                                                         encoder_input_ids=encoder_input_ids,
                                                         attention_mask=attention_mask)
        print(logit_bias)
    if discriminator.model.config.is_encoder_decoder:
        texts = tokenizer.batch_decode(encoder_input_ids, skip_special_tokens=True)
        gen_texts = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        texts = f"{texts}\n\n{gen_texts}"
    else:
        texts = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)

    if openai_model.model_name == 'gpt-3.5-turbo':  # chatgpt
        texts = [format_chat_prompt(t) for t in texts]

    logprobs, indices, usages = parse_logp(openai_model,
                                           tokenizer,
                                           texts,
                                           parameters,
                                           logit_bias=logit_bias)  # [B, k]
    if logprobs.shape[1] == 1:
        return indices.to(decoded_ids.device), usages

    if (use_logit_bias and not pre_post_guidance):
        max_idx = logprobs.argmax(dim=-1)  # [B,]
        max_logit = indices[range(indices.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]
        return max_logit.to(decoded_ids.device), usages

    logprobs = logprobs.to(decoded_ids.device)  # [B, k]
    indices = indices.to(decoded_ids.device)  # [B, k]
    topk_logits = indices.T  # [k, B]
    k = topk_logits.shape[0]

    class_logprob = discriminator(decoded_ids,
                                  topk_logits,
                                  encoder_input_ids=encoder_input_ids,
                                  attention_mask=attention_mask)  # [K * B,]
    class_logprob = class_logprob.view(k, -1).T  # [K, B] => [B, K]
    fudge_topk = logprobs + class_logprob  # [B, K,]
    max_idx = fudge_topk.argmax(dim=-1)  # [B,]
    max_logit = indices[range(indices.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]
    return max_logit, usages


def parse_logp(openai_model, tokenizer, input_texts, parameters, logit_bias=None):
    token_usages = []
    logprobs = []
    indices = []
    for i, t in enumerate(input_texts):
        if logit_bias is not None:
            parameters.logit_bias = logit_bias[i]
        else:
            parameters.logit_bias = {}
        outputs = try_openai_requests(openai_model, t, parameters)
        if outputs is None:
            print("OpenAI API failed. Likely due to rate limit.")
            exit(1)
        token_usages.append(outputs['usage']['total_tokens'])
        if "message" in outputs['choices'][0]:  # chatgpt
            selected_token = outputs['choices'][0]['message']['content']
            logprobs.append([0])  # this means 1 after exp
            encoded = tokenizer.encode(selected_token)
            # if multi-token, only use the first token
            indices.append([encoded[0]])
            continue
        try:
            if len(outputs['choices'][0]['logprobs']['tokens']) == 0:
                selected_token = None
                logp = {100: -float('inf') for _ in range(6)}
            else:
                selected_token = outputs['choices'][0]['logprobs']['tokens'][0]  # str
                selected_logp = outputs['choices'][0]['logprobs']['token_logprobs'][0]  # float
                logp = outputs['choices'][0]['logprobs']["top_logprobs"][0]  # dict
        except Exception as err:
            print(outputs)
            raise err
        if selected_token is not None:
            logp[selected_token] = selected_logp

        prob = []
        idx = []
        for k, v in logp.items():
            if k.startswith('bytes'):
                try:
                    k = eval("b'" + k[6:] + "'").decode('utf-16')
                except:
                    # NOTE: this is a hack to deal with some weird unicode
                    prob.append(-float('inf'))
                    idx.append(101)
            encoded = tokenizer.encode(k)
            # if len(encoded) > 1:
            #     # NOTE: p50kbase tokenizer (gpt3.5 tokenizer) differs from
            #     # gpt2 tokenizer just by a few whitespace tokens.
            #     assert " " * len(k) == k
            idx.append(encoded[0])
            prob.append(v)
        if len(prob) < 6:
            prob.append(-float('inf'))
            idx.append(100)  # in gpt2 (gpt3) tokenizer, 100 is a unicode token often not used
        logprobs.append(prob)
        indices.append(idx)

    return torch.tensor(logprobs), torch.LongTensor(indices), torch.tensor(token_usages)


def try_openai_requests(openai_model, prompt, parameters):
    outputs = None
    try_count = 0
    start = time.time()
    while outputs is None:
        if try_count > 5:
            print(f"Stop trying after {try_count} tries and" f" {time.time() - start:.2f} seconds.")
            return None
        try:
            try_count += 1
            _, outputs = openai_model(prompt, parameters)
        except:
            print("OpenAI Rate Limit reached. Sleeping for 5 minutes.")
            time.sleep(300)
    if try_count > 1:
        print(f"exited while loop after {try_count} tries and"
              f" {time.time() - start:.2f} seconds")

    return outputs


def format_chat_prompt(text):
    prompt = []
    history_text, knowledge_text, instruction, gen_texts = text.split('\n\n')
    knowledge_text = knowledge_text.replace('Knowledge:\n', '')
    dialog_history = history_text.split('\n')[1:]
    for i, utt in enumerate(dialog_history):
        if len(dialog_history) % 2 == 0:
            role = 'assistant' if i % 2 == 0 else 'user'
        else:
            role = 'user' if i % 2 == 0 else 'assistant'
        prompt.append({'role': role, 'content': utt})
    if gen_texts:
        prompt.append({'role': 'assistant', 'content': gen_texts})
        prompt.insert(
            0, {
                'role': 'system',
                'content': f"Be concise. Use the knowledge: \"{knowledge_text}\"."
                           "Complete the assistant utterance."
            })
    else:
        prompt.insert(0, {'role': 'system', 'content': f"Use the knowledge: \"{knowledge_text}\"."})
    return prompt
