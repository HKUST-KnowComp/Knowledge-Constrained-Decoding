from dataclasses import dataclass
import os
import random

from transformers import HfArgumentParser

from kcd.kilt.load_kilt_data import load_fever
from datasets import concatenate_datasets, Dataset


@dataclass
class ExperimentArgs:
    dataset_name: str = 'wow'
    dataset_path: str = 'data/wow.jsonl'
    use_kilt_format: bool = False
    sample_method: str = 'random'  # choices: [random]


def main():
    parser = HfArgumentParser([ExperimentArgs])
    args = parser.parse_args_into_dataclasses()[0]

    if args.dataset_name == 'wow':
        if args.use_kilt_format:
            from kcd.kilt.load_kilt_data import load_wow
        else:
            from kcd.wizard_of_wikipedia import load_wow
        data_load_fn = load_wow
    else:
        data_load_fn = load_fever

    dataset, config = data_load_fn(args.dataset_path)

    ans = dataset['answers']
    knowledge = dataset['ctxs']
    if args.sample_method == 'random':
        random.shuffle(knowledge)
        neg_dataset = Dataset.from_dict({
            'answers': ans,
            'ctxs': knowledge,
            'question': dataset['question'],
            'label': [0 for _ in range(len(ans))],
        })
    else:
        raise ValueError
    full_data = concatenate_datasets([dataset, neg_dataset])
    basename = os.path.basename(args.dataset_path).split('.')[0]
    dataset_name = f'{args.dataset_name}_{basename}_augmented_neg_{args.sample_method}'
    full_data.save_to_disk(f'data/cached/{dataset_name}')


if __name__ == '__main__':
    main()
