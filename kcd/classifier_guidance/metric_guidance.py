"""
Instead of classifiers, use metrics to score and re-rank tokens.
"""
import torch

from .utils import zero_out_after_eos

class MetricGuidance:
    """
    Metric guidance class to be used with PplMCTS.
    """

    def __init__(self,
                 tokenizer,
                 metric,
                 metric_name=None,
                 max_new_tokens: int = 32,
                 k: int = 200) -> None:
        self.tokenizer = tokenizer
        self.metric = metric
        self.metric_name = metric_name
        self.max_new_tokens = max_new_tokens
        self.k = k

    def __call__(self, reference_ids, decoded_ids):
        """
        reference_ids: torch.LongTensor of shape [B, T_1]
        decoder_ids: torch.LongTensor of shape [B, T_2]
        """
        reference = self.tokenizer.batch_decode(reference_ids, skip_special_tokens=True)
        decoded = self.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        scores = self.metric(decoded, [[ref] for ref in reference])
        if isinstance(scores, dict):
            assert self.metric_name is not None
            scores = scores[self.metric_name]
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)  # [B,]
        return scores


def metric_guided_generation(metric=None, metric_name=None, max_new_tokens: int = 32, k: int = 200):
    assert metric is not None

    def _generate_fn(model, tokenizer, inputs):
        """input_ids: torch.LongTensor, knowledge_text: str"""
        input_ids = inputs['input_ids']
        knowledge_text = tokenizer.batch_decode(inputs['knowledge_ids'], skip_special_tokens=True)

        generated_tokens = torch.LongTensor([[model.config.decoder_start_token_id]
                                            ]).to(input_ids.device)
        generated_tokens = generated_tokens.repeat(input_ids.shape[0], 1)
        for i in range(max_new_tokens):
            next_logit = metric_guidance_step(model,
                                              tokenizer,
                                              metric,
                                              generated_tokens,
                                              knowledge_text,
                                              gen_inst_ids=input_ids,
                                              k=k,
                                              metric_name=metric_name)
            generated_tokens = torch.cat([generated_tokens, next_logit], dim=1)
            # early stopping based on eos
            if (generated_tokens == tokenizer.eos_token_id).any(dim=1).all():
                break
        return zero_out_after_eos(generated_tokens, tokenizer.eos_token_id)

    return _generate_fn


# the metric can be TokenF1Score.batch_compute
@torch.inference_mode()
def metric_guidance_step(model,
                         tokenizer,
                         metric,
                         decoded_ids: torch.LongTensor,
                         knowledge_text: list[str],
                         gen_inst_ids: torch.LongTensor = None,
                         k: int = 200,
                         metric_name=None):
    if model.config.is_encoder_decoder:
        assert gen_inst_ids is not None
        input_ids = gen_inst_ids
        logits = model(input_ids=input_ids, decoder_input_ids=decoded_ids).logits  # [1, T, V]
    else:
        input_ids = decoded_ids
        if gen_inst_ids is not None:
            input_ids = torch.cat([gen_inst_ids, input_ids], dim=1)
        else:
            input_ids = decoded_ids
        logits = model(input_ids=input_ids).logits
    next_logit = torch.softmax(logits[:, -1, :], dim=-1)  # [B, V]
    values, indices = next_logit.topk(k, dim=-1)  # [B, k]
    topk_logits = indices.T  # [k, B]

    candidates = []
    for idx in topk_logits:
        curr = torch.cat([decoded_ids[:, 1:], idx.unsqueeze(1)], dim=1)
        candidates.append(curr)

    candidates = torch.stack(candidates, dim=0)  # [K, B, T]
    candidates = candidates.view(-1, candidates.shape[-1])  # [K * B, T]
    candidate_str = tokenizer.batch_decode(candidates, skip_special_tokens=True)  # [K * B,]

    scores = metric(candidate_str, [[kt] for kt in knowledge_text] * k)
    if isinstance(scores, dict):
        scores = scores[metric_name]
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores).to(values.device)  # [K * B,]

    reranked = values * scores.view(k, -1).T  # [B, K]
    max_idx = reranked.argmax(dim=-1)  # [B,]
    max_logit = indices[range(max_idx.shape[0]), max_idx].unsqueeze(-1)  # [B, 1]
    return max_logit
