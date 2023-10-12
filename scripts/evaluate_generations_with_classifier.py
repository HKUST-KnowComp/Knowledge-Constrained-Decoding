from dataclasses import dataclass, field
import glob
import json
import os

from datasets import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

from kcd.token_classifier.dataloader import DataCollatorForSeq2SeqTokenClassification, load_data
from kcd.token_classifier.model import T5DoubleHeadModel
from kcd.util import shift_right

@dataclass
class ExperimentArgs:
    model_name: str = field(default="google/flan-t5-xl")
    num_labels: int = field(default=2)
    attr_idx: int = 1
    load_8bit: bool = True
    instruction_model: str = 'basic'  # 'basic' or 'alpaca'
    dataset: str = 'wow'  # 'wow' or 'fever'
    use_kilt_format: bool = False
    test_data_path: str = "data/cached/wow/test_unseen.jsonl"
    generations_path: str = "generations/pplm_prompts.jsonl"
    causal_lm_generations: bool = False
    load_checkpoint: str = None
    load_peft_checkpoint: str = None
    load_classifier: str = field(default=None)
    use_mlp_classifier: bool = False
    batch_size: int = 1
    skip_no_knowledge: bool = False


def main():
    parser = HfArgumentParser([ExperimentArgs])
    args = parser.parse_args_into_dataclasses()[0]
    args.train_data_path = None
    args.validation_data_path = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_kwargs = {
        'device_map': 'auto' if args.load_8bit else None,
        'load_in_8bit': args.load_8bit,
        'torch_dtype': torch.float16 if args.load_8bit else torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kwargs)
    tokenizer.truncation_side = 'left'

    if args.use_mlp_classifier:
        load_kwargs.pop('load_in_8bit')
        load_kwargs['use_mlp_classifier'] = True
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=args.load_8bit,
                                                                llm_int8_skip_modules=['lm_head', 'mlp_layer1', 'mlp_layer2'])
    model = T5DoubleHeadModel.from_pretrained(args.model_name,
                                           output_hidden_states=True,
                                           use_cache=True,
                                           num_labels=args.num_labels,
                                           pool_method='last',
                                           **load_kwargs)

    if args.load_peft_checkpoint:
        model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)

    if args.load_checkpoint:
        peft_config_path = os.path.join(os.path.dirname(args.load_checkpoint), 'adapter_model')
        peft_config = LoraConfig.from_pretrained(peft_config_path)
        model = get_peft_model(model, peft_config)
        incompatible = model.load_state_dict(torch.load(args.load_checkpoint), strict=False)
        assert (len(incompatible.missing_keys) == 1
                and incompatible.missing_keys[0].endswith('lm_head.weight'))

    if args.load_classifier:
        ckpt = torch.load(args.load_classifier)
        ckpt = {k.replace('classifier.', ''): v for k, v in ckpt.items() if 'classifier' in k}
        model.classifier.load_state_dict(ckpt, strict=True)

    dataset = load_data(args,
                        tokenizer,
                        is_encoder_decoder=model.config.is_encoder_decoder,
                        instruction_model=args.instruction_model)['test']

    if '*' in args.generations_path:
        paths = glob.glob(args.generations_path)
    elif os.path.isdir(args.generations_path):
        paths = glob.glob(os.path.join(args.generations_path, '*.jsonl'))
    else:
        paths = [args.generations_path]

    outfile = open('class_prob.jsonl', 'a')
    for path in paths:
        print(f"loading generations at {path}...")
        gen_dataset = load_generations(path,
                                       dataset,
                                       tokenizer,
                                       causal_lm_generations=args.causal_lm_generations)
        if args.skip_no_knowledge:
            gen_dataset = gen_dataset.filter(
                lambda x: 'no_passages_used' not in tokenizer.decode(x['input_ids']))
        collator = DataCollatorForSeq2SeqTokenClassification(tokenizer)
        dataloader = DataLoader(gen_dataset,
                                batch_size=args.batch_size,
                                collate_fn=collator,
                                shuffle=False)
        print("generations loaded")
        samples_pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        sum_probs = 0
        for i, batch in samples_pbar:
            batch = batch.to(device)
            _, output = model(**batch)
            probs = torch.softmax(output.logits, dim=-1)[:, args.attr_idx]  # [B,]
            sum_probs += probs.sum().item()

        mean_score = sum_probs / len(gen_dataset)
        print(f"Evaluation of {path}: {mean_score}")
        outfile.write(json.dumps({'path': path, 'score': mean_score}) + '\n')
    outfile.close()


def load_generations(generations_path, dataset, tokenizer, causal_lm_generations=False):
    df = pd.read_json(generations_path, lines=True)
    data_df = dataset.to_pandas()
    data_df = data_df.iloc[:len(df)]  # truncate if generations are shorter
    df['labels'] = data_df['labels'].apply(lambda x: x[0])
    gen_dataset = Dataset.from_pandas(df)
    def _tokenize(example):
        if isinstance(example['response'], dict):
            # for chat GPT
            if 'text' in example['response']['choices'][0]:
                gen = example['response']['choices'][0]['text']
            else:
                gen = example['response']['choices'][0]['message']['content']
        else:
            gen = example['response']
        if causal_lm_generations:
            try:
                gen = gen.split('### Response:')[1]
            except:
                print("No response found in generation.")
                gen = "no response"
        gen_ids = tokenizer(gen,
                            truncation=True,
                            max_length=tokenizer.model_max_length,
                            return_tensors='pt').input_ids[0]
        gen_ids = shift_right(gen_ids, tokenizer.pad_token_id)
        labels = torch.full_like(gen_ids, example['labels'])
        return {'decoder_input_ids': gen_ids, 'labels': labels}

    tokenized = gen_dataset.map(_tokenize, remove_columns=gen_dataset.column_names)
    data_df['decoder_input_ids'] = tokenized['decoder_input_ids']
    data_df['labels'] = tokenized['labels']
    dataset = Dataset.from_pandas(data_df)
    return dataset


if __name__ == "__main__":
    main()
