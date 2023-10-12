from copy import deepcopy
from functools import partial
import json
import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

from kcd.evaluation.auto_evaluation import evaluate_per_sent
from kcd.evaluation.token_f1_score import TokenF1Score

from kcd.token_classifier.dataloader import load_data, DataCollatorForSeq2SeqTokenClassification
from kcd.token_classifier.model import T5DoubleHeadModel
from kcd.classifier_guidance.ppl_mcts import PplMCTS, PplMCTSConfig
from kcd.configs import GenerationConfig
from kcd.classifier_guidance.metric_guidance import MetricGuidance


@dataclass
class ExperimentArgs:
    lm_name: str = field(default="google/flan-t5-xl")
    num_labels: int = field(default=2)
    attr_idx: int = 1
    dataset: str = 'wow'
    use_kilt_format: bool = True
    load_8bit: bool = True
    test_data_path: str = field(default="data/pplm_prompts.csv")
    output_path: str = 'generations/ppl_mcts'
    disc_name: str = ''
    instruction_model: str = 'basic'  # choices=['basic', 'openai', 'alpaca']
    guide_using_metric: bool = False
    metric_name: str = 'token_f1'
    load_peft_checkpoint: str = None
    load_checkpoint: str = None
    load_classifier: str = field(default=None)
    use_mlp_classifier: bool = False
    v2: bool = False
    batch_size: int = 1
    continue_from: int = 0
    human_indices: str = None

    guide_after: int = 0
    complete_after: int = 0
    guide_every: int = 1
    guide_for: int = 0

    max_num_gen: int = -1


def main():
    parser = HfArgumentParser([ExperimentArgs, PplMCTSConfig, GenerationConfig])
    args, mcts_args, gen_cfg = parser.parse_args_into_dataclasses()
    args.train_data_path = None
    args.validation_data_path = None

    print('cuda available:', torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_kwargs = {
        'device_map': 'auto',
        'load_in_8bit': args.load_8bit,
        'torch_dtype': torch.float16,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name, **load_kwargs)
    # tokenizer.padding_side = "left"  # only for causal LM
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_mlp_classifier:
        load_kwargs.pop('load_in_8bit')
        load_kwargs['use_mlp_classifier'] = True
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=args.load_8bit,
                                                                llm_int8_skip_modules=['lm_head', 'mlp_layer1', 'mlp_layer2'])

    lm = T5DoubleHeadModel.from_pretrained(args.lm_name,
                                           output_hidden_states=True,
                                           use_cache=True,
                                           num_labels=args.num_labels,
                                           v2=args.v2,
                                           **load_kwargs)
    if args.guide_using_metric:
        if args.metric_name == 'token_f1':
            metric = TokenF1Score.batch_compute
        elif args.metric_name in ('bleu', 'bertscore', 'rougeL', 'weighted_bleu'):
            if args.metric_name == 'weighted_bleu':
                # NOTE: idea - negative weight for higher n-gram -> less copy
                bleu_weights = (2, 1, -0.5, -1.5)
                metric = partial(evaluate_per_sent, metric='bleu', bleu_weights=bleu_weights)
            else:
                metric = partial(evaluate_per_sent, metric=args.metric_name)
        else:
            raise NotImplementedError(f"Metric {args.metric_name} not implemented")
        classi = MetricGuidance(tokenizer,
                                metric,
                                metric_name=args.metric_name,
                                max_new_tokens=gen_cfg.max_new_tokens,
                                k=gen_cfg.top_k)
    else:
        if args.load_peft_checkpoint:
            lm = PeftModel.from_pretrained(lm, args.load_peft_checkpoint)
        if args.load_checkpoint:
            peft_config_path = os.path.join(os.path.dirname(args.load_checkpoint), 'adapter_model')
            peft_config = LoraConfig.from_pretrained(peft_config_path)
            lm = get_peft_model(lm, peft_config)
            incompatible = lm.load_state_dict(torch.load(args.load_checkpoint), strict=False)
            assert (len(incompatible.missing_keys) == 1 and
                    incompatible.missing_keys[0].endswith('lm_head.weight'))
        if args.load_classifier:
            ckpt = torch.load(args.load_classifier)
            ckpt = {k.replace('classifier.', ''): v for k, v in ckpt.items() if 'classifier' in k}
            lm.classifier.load_state_dict(ckpt, strict=True)

        def classi(**kwargs):
            lm_output, class_output = lm(**kwargs)
            lm_output.logits = class_output.logits
            return lm_output

    print("loading dataset")
    dataset = load_data(args,
                        tokenizer,
                        is_encoder_decoder=lm.config.is_encoder_decoder,
                        instruction_model=args.instruction_model,
                        get_knowledge_ids=True)['test']
    indices = None
    if args.max_num_gen > 0:
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), args.max_num_gen)
        dataset = dataset.select(indices)

    if args.human_indices:
        with open(args.human_indices) as f:
            indices = [int(i) for i in f.readlines()]
        dataset = dataset.select(indices)
    if args.continue_from > 0:
        dataset = dataset.select(range(args.continue_from, len(dataset)))
    collator = DataCollatorForSeq2SeqTokenClassification(tokenizer,
                                                         other_features_to_pad=['knowledge_ids'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    print("dataset loaded")
    batch_size = args.batch_size
    MCTS = PplMCTS(mcts_args,
                   tokenizer,
                   lm,
                   classi,
                   gedi=args.v2,
                   disable_adapter_lm_forward=(not args.guide_using_metric and args.load_classifier is None),
                   batch_size=batch_size,
                   temperature=gen_cfg.temperature,
                   num_labels=args.num_labels,
                   unused_token_id=tokenizer.unk_token_id,
                   device=device)
    samples_pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Samples generated")

    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = args.dataset
    if args.use_kilt_format:
        dataset_name = f'{dataset_name}-kilt'
    if args.complete_after:
        args.disc_name += f'-complete_after{args.complete_after}'
    out_fname = os.path.join(
        args.output_path, f'{dataset_name}-{args.lm_name.replace("/", "-")}-{args.disc_name}.jsonl')
    fout = open(out_fname, 'a')
    for i, batch in samples_pbar:
        gold_ref_ids = batch.pop('decoder_input_ids')
        batch = batch.to(device)
        labels = torch.zeros(batch['input_ids'].shape[0], args.num_labels, device=device)
        labels[:, args.attr_idx] = 1

        if args.guide_after > 0:
            generation_kwargs = deepcopy(gen_cfg.__dict__)
            generation_kwargs['max_new_tokens'] = args.guide_after
            decoder_ids = (lm.config.decoder_start_token_id *
                           torch.ones(batch['input_ids'].shape[0], 1, dtype=torch.long, device=device))

            is_peft = hasattr(lm, 'base_model')
            if is_peft:
                lm.base_model.base_model.return_lm_only = True
                with lm.disable_adapter():
                    generated = lm.generate(input_ids=batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            decoder_input_ids=decoder_ids,
                                            **generation_kwargs)
                lm.base_model.base_model.return_lm_only = False
            else:
                lm.return_lm_only = True
                generated = lm.generate(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        decoder_input_ids=decoder_ids,
                                        **generation_kwargs)
                lm.return_lm_only = False
            inputs = deepcopy(batch)
            inputs['decoder_input_ids'] = generated
            MCTS.set_labels(labels)
            decoded, _ = MCTS.search(inputs,
                                     tokens_to_generate=gen_cfg.max_new_tokens - args.guide_after,
                                     continuing=True)
        elif args.guide_every > 1:
            assert args.guide_for > 0
            num_guide = gen_cfg.max_new_tokens % (args.guide_every + args.guide_for)
            generation_kwargs = deepcopy(gen_cfg.__dict__)
            generation_kwargs['max_new_tokens'] = args.guide_every

            decoder_ids = (lm.config.decoder_start_token_id *
                           torch.ones(batch['input_ids'].shape[0], 1, dtype=torch.long, device=device))
            for _ in range(num_guide):
                is_peft = hasattr(lm, 'base_model')
                if is_peft:
                    lm.base_model.base_model.return_lm_only = True
                    with lm.disable_adapter():
                        decoder_ids = lm.generate(input_ids=batch['input_ids'],
                                                  attention_mask=batch['attention_mask'],
                                                  decoder_input_ids=decoder_ids,
                                                  **generation_kwargs)
                    lm.base_model.base_model.return_lm_only = False
                else:
                    lm.return_lm_only = True
                    generated = lm.generate(input_ids=batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            decoder_input_ids=decoder_ids,
                                            **generation_kwargs)
                    lm.return_lm_only = False
                inputs = deepcopy(batch)
                inputs['decoder_input_ids'] = decoder_ids
                MCTS.set_labels(labels)
                _, decoder_ids = MCTS.search(inputs,
                                             tokens_to_generate=args.guide_for,
                                             continuing=True)
                decoder_ids = torch.stack(decoder_ids, dim=0)
            decoded = torch.batch_decode(decoder_ids, skip_special_tokens=True)
        elif args.complete_after > 0:
            # first, generate with MCTS
            MCTS.set_labels(labels)
            _, grounded_ids = MCTS.search(batch,
                                          tokens_to_generate=args.complete_after)
            grounded_ids = torch.stack(grounded_ids, dim=0).to(device)

            # now, complete with LM
            generation_kwargs = deepcopy(gen_cfg.__dict__)
            generation_kwargs['max_new_tokens'] = gen_cfg.max_new_tokens - args.complete_after

            is_peft = hasattr(lm, 'base_model')
            if is_peft:
                lm.base_model.base_model.return_lm_only = True
                with lm.disable_adapter():
                    generated = lm.generate(input_ids=batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            decoder_input_ids=grounded_ids,
                                            **generation_kwargs)
                lm.base_model.base_model.return_lm_only = False
            else:
                lm.return_lm_only = True
                generated = lm.generate(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        decoder_input_ids=grounded_ids,
                                        **generation_kwargs)
                lm.return_lm_only = False
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        else:
            # MCTS search for whole sequence
            MCTS.set_labels(labels)
            decoded, _ = MCTS.search(batch, tokens_to_generate=gen_cfg.max_new_tokens)

        for j, (prompt_ids, response,
                gold_ids) in enumerate(zip(batch['input_ids'], decoded,
                                           gold_ref_ids)):
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            gold = tokenizer.decode(gold_ids, skip_special_tokens=True)
            result = {
                'prompt': prompt,
                'response': response,
                'gold': gold,
                'index': i * batch_size + j + args.continue_from,
            }
            if indices is not None:
                result['index'] = indices[i * batch_size + j + args.continue_from]
            fout.write(json.dumps(result) + '\n')
    fout.close()


if __name__ == "__main__":
    main()
