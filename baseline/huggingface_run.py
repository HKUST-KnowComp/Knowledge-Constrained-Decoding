import json
import os
from dataclasses import dataclass
from pprint import pprint

import torch
from tqdm.auto import tqdm
from transformers import (DataCollatorForSeq2Seq, HfArgumentParser, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model

from kcd.text_data import load_text_data
from kcd.util import load_transformer_LM_tokenizer
from kcd.configs import GenerationConfig


@dataclass
class ExperimentArgs:
    data_path: str = 'data/wow-dev-kilt-processed.jsonl'
    output_path: str = 'generations/baseline'
    model_name: str = "google/flan-t5-xl"
    dataset: str = 'wow'
    use_kilt_format: bool = True
    task: str = 'completion'
    continue_from: int = 0
    batch_size: int = 1
    load_8bit: bool = True
    load_checkpoint: str = ''
    print_output: bool = False
    instruction_model: str = 'basic'  # choices=['basic', 'openai', 'alpaca']


def main():
    parser = HfArgumentParser((ExperimentArgs, GenerationConfig, Seq2SeqTrainingArguments))
    args, gen_cfg, train_args = parser.parse_args_into_dataclasses()
    args.output_path = train_args.output_dir

    load_kwargs = {
        'device_map': 'auto' if args.load_8bit else None,
        'load_in_8bit': args.load_8bit,
        'torch_dtype': torch.float16 if args.load_8bit else torch.bfloat16,
    }

    model, tokenizer = load_transformer_LM_tokenizer(args.model_name, **load_kwargs)
    tokenizer.truncation_side = 'left'

    if args.load_checkpoint:
        peft_config_path = os.path.join(os.path.dirname(args.load_checkpoint), 'adapter_model')
        peft_config = LoraConfig.from_pretrained(peft_config_path)
        model = get_peft_model(model, peft_config)

        ckpt = torch.load(args.load_checkpoint)
        ckpt['base_model.model.lm_head.weight'] = ckpt.pop("base_model.model.lm_head.0.weight")
        model.load_state_dict(ckpt, strict=True)

    tokenized_dataset = load_text_data(path=args.data_path,
                                       instruction_model=args.instruction_model,
                                       task=args.task,
                                       use_kilt_format=args.use_kilt_format,
                                       tokenize=True,
                                       tokenizer=tokenizer,
                                       no_label=True)
    text_dataset = load_text_data(path=args.data_path,
                                  instruction_model=args.instruction_model,
                                  task=args.task,
                                  use_kilt_format=args.use_kilt_format,
                                  tokenize=False)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
    )
    preds = trainer.predict(tokenized_dataset, **gen_cfg.__dict__)
    preds.predictions[preds.predictions == -100] = tokenizer.pad_token_id
    responses = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = args.dataset
    if args.use_kilt_format:
        dataset_name += '-kilt'
    out_fname = f'{dataset_name}-{args.model_name.replace("/", "-")}'
    if args.load_checkpoint:
        out_fname = out_fname + '-sft'
    out_fname = os.path.join(args.output_path, out_fname + '.jsonl')
    fout = open(out_fname, 'a')
    for i, (example, response) in tqdm(enumerate(zip(text_dataset, responses)),
                                       total=len(responses)):
        result = {
            'prompt': example['prompt'],
            'response': response,
            'gold': example['completion'],
            'index': i,
        }
        fout.write(json.dumps(result) + '\n')
        if args.print_output:
            print("prompt:\n")
            pprint(example['prompt'])
            print()
            print(f"{args.model_name} Response:\n")
            print(response + '\n\n')
            print(f"Gold Response:\n")
            print(example['completion'] + '\n\n')
    fout.close()


if __name__ == "__main__":
    main()
