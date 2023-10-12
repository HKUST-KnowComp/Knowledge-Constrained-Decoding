from dataclasses import dataclass, field
import glob
import os

from datasets import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, DataCollatorWithPadding

from kcd.token_classifier.dataloader import load_data

@dataclass
class ExperimentArgs:
    instruction_model: str = 'basic'  # 'basic' or 'alpaca'
    dataset: str = 'wow'  # 'wow' or 'fever'
    use_kilt_format: bool = False
    test_data_path: str = "data/cached/wow/test_unseen.jsonl"
    generations_path: str = "generations/pplm_prompts.jsonl"
    causal_lm_generations: bool = False
    batch_size: int = 1


def main():
    parser = HfArgumentParser([ExperimentArgs])
    args = parser.parse_args_into_dataclasses()[0]
    args.train_data_path = None
    args.validation_data_path = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("henry931007/mfma")
    model = AutoModelForSequenceClassification.from_pretrained("henry931007/mfma").to(device)

    dataset = load_data(args,
                        tokenizer,
                        is_encoder_decoder=model.config.is_encoder_decoder,
                        instruction_model=args.instruction_model,
                        get_knowledge_ids=True)['test']

    if '*' in args.generations_path:
        paths = glob.glob(args.generations_path)
    elif os.path.isdir(args.generations_path):
        paths = glob.glob(os.path.join(args.generations_path, '*.jsonl'))
    else:
        paths = [args.generations_path]
    for path in paths:
        print(f"loading generations at {path}...")
        gen_dataset = load_generations(path,
                                       dataset,
                                       tokenizer,
                                       causal_lm_generations=args.causal_lm_generations)
        collator = DataCollatorWithPadding(tokenizer)
        dataloader = DataLoader(gen_dataset,
                                batch_size=args.batch_size,
                                collate_fn=collator,
                                shuffle=False)
        print("generations loaded")
        samples_pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        sum_probs = 0
        for i, batch in samples_pbar:
            batch = batch.to(device)
            output = model(**batch)
            probs = torch.softmax(output.logits, dim=-1)[:, 0]  # [B,]
            sum_probs += probs.sum().item()

        mean_score = sum_probs / len(gen_dataset)
        print(f"Evaluation of {path}: {mean_score}")


def load_generations(generations_path, dataset, tokenizer, causal_lm_generations=False):
    df = pd.read_json(generations_path, lines=True)
    data_df = dataset.to_pandas()
    data_df = data_df.iloc[:len(df)]  # truncate if generations are shorter
    df['labels'] = data_df['labels'].apply(lambda x: x[0])
    df['knowledge_ids'] = data_df['knowledge_ids'].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
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

        inputs = tokenizer(example['knowledge_ids'],
                           gen,
                           truncation=True,
                           max_length=tokenizer.model_max_length,
                           return_tensors='pt')
        inputs['labels'] = torch.LongTensor([example['labels']])
        return {k: v[0] for k, v in inputs.items()}

    tokenized = gen_dataset.map(_tokenize, remove_columns=gen_dataset.column_names)
    return tokenized


if __name__ == "__main__":
    main()
