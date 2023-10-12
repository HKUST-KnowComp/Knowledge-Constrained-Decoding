import os
import random

from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer

MAIN_MODEL = 'google/flan-t5-xl'


def load_summary_data(path: str, tokenizer=None, split='test', max_train_samples=100000, random_sample=False):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL)
    config = {
        'input_columns': ['ctxs'],
        'instruction': "### Document:\n{}"
                       "\n\nGiven the article above, generate a faithful summary."
    }
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        return dataset, config
    if path == 'cnn_dailymail':
        dataset = load_dataset(path, '3.0.0')[split]
        if split == 'train':
            indices = list(range(len(dataset)))
            if random_sample:
                random.shuffle(indices)
            dataset = dataset.select(indices[:max_train_samples])
        dataset = dataset.rename_column('article', 'ctxs')
        dataset = dataset.rename_column('highlights', 'question')
        dataset = dataset.add_column('answers', dataset['question'])
    elif path == 'xsum':
        dataset = load_dataset(path)[split]
        dataset = dataset.rename_column('document', 'ctxs')
        dataset = dataset.rename_column('summary', 'question')
        dataset = dataset.add_column('answers', dataset['question'])
    else:
        raise ValueError(f'Unknown dataset: {path}')
    dataset = dataset.add_column('label', [1] * len(dataset))

    # Tokenize the dataset and filter out samples that are too long
    def get_doc_len(examples):
        return {'doc_len': len(tokenizer.encode(examples['ctxs']))}

    dataset = dataset.map(get_doc_len)
    # -25 for instructions
    dataset = dataset.filter(lambda x: x['doc_len'] <= tokenizer.model_max_length - 25)

    return dataset, config
