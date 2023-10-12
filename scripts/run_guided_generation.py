from functools import partial
import json
import os
from dataclasses import dataclass
from pprint import pprint

import torch
from tqdm.auto import tqdm
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

from kcd.token_classifier.dataloader import load_data, DataCollatorForSeq2SeqTokenClassification
from kcd.classifier_guidance import GuidedGenerationPredictor, load_generate_fn
from kcd.evaluation import TokenF1Score, evaluate_per_sent
from kcd.util import load_transformer_LM_tokenizer
from kcd.configs import GenerationConfig


@dataclass
class ExperimentArgs:
    test_data_path: str = 'data/wow-dev-kilt-processed.jsonl'
    output_path: str = 'generations/fudge'
    model_name: str = "google/flan-t5-xl"
    dataset: str = 'wow'
    use_kilt_format: bool = True
    load_8bit: bool = True
    print_output: bool = False
    instruction_model: str = 'basic'  # choices=['basic', 'openai', 'alpaca']
    guidance_method: str = 'fudge'  # choices=['metric_guidance', 'fudge', 'nado']
    metric: str = 'token_f1'  # choices=['token_f1']
    disc_name: str = ''

    num_labels: int = 2
    load_checkpoint: str = None
    load_peft_checkpoint: str = None
    load_classifier: str = None
    use_mlp_classifier: bool = False
    continue_from: int = 0
    v2: bool = False
    human_indices: str = None
    alpha: float = 1.0  # how much grounded for nado

    complete_after: int = 0


def main():
    parser = HfArgumentParser((ExperimentArgs, GenerationConfig, Seq2SeqTrainingArguments))
    args, gen_cfg, train_args = parser.parse_args_into_dataclasses()
    args.output_path = train_args.output_dir
    train_args.predict_with_generate = True
    train_args.remove_unused_columns = False  # keep to False
    args.train_data_path = None
    args.validation_data_path = None

    load_kwargs = {
        'device_map': 'auto' if args.load_8bit else None,
        'load_in_8bit': args.load_8bit,
        'torch_dtype': torch.float16 if args.load_8bit else torch.bfloat16,
    }
    if args.guidance_method in ('fudge', 'nado', 'astar'):
        load_kwargs['num_labels'] = args.num_labels
        load_kwargs['pool_method'] = 'last'  # for efficiency
        load_kwargs['load_t5_doublehead'] = True
        load_kwargs['v2'] = args.v2
    print('cuda available:', torch.cuda.is_available())

    if args.use_mlp_classifier:
        load_kwargs.pop('load_in_8bit')
        load_kwargs['use_mlp_classifier'] = True
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=args.load_8bit,
                                                                llm_int8_skip_modules=['lm_head', 'mlp_layer1', 'mlp_layer2'])

    model, tokenizer = load_transformer_LM_tokenizer(args.model_name, **load_kwargs)

    if args.load_peft_checkpoint:
        model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)
    if args.load_checkpoint:
        peft_config_path = os.path.join(os.path.dirname(args.load_checkpoint), 'adapter_model')
        peft_config = LoraConfig.from_pretrained(peft_config_path)
        model = get_peft_model(model, peft_config)
        incompatible = model.load_state_dict(torch.load(args.load_checkpoint), strict=False)
        assert (len(incompatible.missing_keys) == 1 and
                incompatible.missing_keys[0].endswith('lm_head.weight'))
    if args.load_classifier:
        ckpt = torch.load(args.load_classifier)
        ckpt = {k.replace('classifier.', ''): v for k, v in ckpt.items() if 'classifier' in k}
        model.classifier.load_state_dict(ckpt, strict=True)

    if not model.config.is_encoder_decoder:
        raise NotImplementedError(
            'The dataloading for non-encoder-decoder is not set yet for inference.'
            'Take a look at kcd.token_classifier.dataloader.')

    dataset = load_data(args,
                        tokenizer,
                        is_encoder_decoder=model.config.is_encoder_decoder,
                        instruction_model=args.instruction_model,
                        get_knowledge_ids=True)['test']
    indices = None
    if args.human_indices:
        with open(args.human_indices) as f:
            indices = [int(i) for i in f.readlines()]
        dataset = dataset.select(indices)
    if args.continue_from > 0:
        dataset = dataset.select(range(args.continue_from, len(dataset)))
    # load guidance criteria
    if args.guidance_method == 'metric_guidance':
        if args.metric == 'token_f1':
            metric = TokenF1Score.batch_compute
        elif args.metric in ('bleu', 'bertscore', 'rougeL', 'weighted_bleu'):
            if args.metric == 'weighted_bleu':
                # NOTE: idea - negative weight for higher n-gram -> less copy
                # bleu_weights = (2, 1, -0.5, -1.5)
                bleu_weights = (0.5, 0.25, 0.2, 0.05)
                metric = partial(evaluate_per_sent, metric='bleu', bleu_weights=bleu_weights)
            else:
                metric = partial(evaluate_per_sent, metric=args.metric)
        else:
            raise NotImplementedError(f"Metric {args.metric} not implemented")
        generate_fn = load_generate_fn(args.guidance_method,
                                       metric=metric,
                                       metric_name=args.metric,
                                       max_new_tokens=gen_cfg.max_new_tokens,
                                       k=gen_cfg.top_k)

    elif args.guidance_method == 'fudge':
        generate_fn = load_generate_fn(args.guidance_method,
                                       model=model,
                                       max_new_tokens=gen_cfg.max_new_tokens,
                                       k=gen_cfg.top_k,
                                       complete_after=args.complete_after,
                                       disable_adapter_lm_forward=args.load_classifier is None)
    elif args.guidance_method == 'nado':
        generate_fn = load_generate_fn(args.guidance_method,
                                       model=model,
                                       max_new_tokens=gen_cfg.max_new_tokens,
                                       k=gen_cfg.top_k,
                                       alpha=args.alpha,
                                       disable_adapter_lm_forward=args.load_classifier is None)
    elif args.guidance_method == 'astar':
        generate_fn = load_generate_fn(args.guidance_method,
                                       model=model,
                                       max_new_tokens=gen_cfg.max_new_tokens,
                                       k=gen_cfg.top_k,
                                       disable_adapter_lm_forward=args.load_classifier is None,
                                       future_steps=5,
                                       lambda_weight=0.25,
                                       soft_forward=False)

    else:
        raise NotImplementedError(f"Guidance method {args.guidance_method} not implemented")

    trainer = GuidedGenerationPredictor(
        generate_fn=generate_fn,
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2SeqTokenClassification(
            tokenizer,
            other_features_to_pad=['knowledge_ids'],
        ),
        tokenizer=tokenizer,
    )
    preds = trainer.predict(dataset, **gen_cfg.__dict__)
    preds.predictions[preds.predictions == -100] = tokenizer.pad_token_id
    responses = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = args.dataset
    if args.use_kilt_format:
        dataset_name = f'{dataset_name}-kilt'

    base_fname = args.model_name.replace("/", "-")
    if args.guidance_method == 'metric_guidance':
        base_fname += f'-{args.metric}'
    else:
        base_fname += f'-{args.guidance_method}-{args.disc_name}'
    out_fname = os.path.join(args.output_path, f'{dataset_name}-{base_fname}.jsonl')
    fout = open(out_fname, 'a')
    for i, (example, response) in tqdm(enumerate(zip(dataset, responses)), total=len(responses)):
        prompt = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        completion = tokenizer.decode(example['decoder_input_ids'], skip_special_tokens=True)
        result = {
            'prompt': prompt,
            'response': response,
            'gold': completion,
            'index': i + args.continue_from,
        }
        if indices is not None:
            result['index'] = indices[i]
        fout.write(json.dumps(result) + '\n')
        if args.print_output:
            print("prompt:\n")
            pprint(prompt)
            print()
            print(f"{args.model_name} Response:\n")
            print(response + '\n\n')
            print(f"Gold Response:\n")
            print(completion + '\n\n')
    fout.close()


if __name__ == "__main__":
    main()
