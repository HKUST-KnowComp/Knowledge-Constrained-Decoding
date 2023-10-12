import os
import random

from datasets import Dataset, load_from_disk
import pandas as pd


def load_wow(path: str, max_samples=None, random_sample=False):
    wow_config = {
        'input_columns': ['ctxs', 'question'],
        'instruction': "History:\n{}\n\nKnowledge:\n{}"
                       "\n\nGiven the dialog history and a relevant knowledge above,"
                       " generate a knowledgeable, usefule, and helpful answer."
    }
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        return dataset, wow_config

    df = pd.read_json(path, lines=True)
    df['question'] = df['history'].apply(lambda x: '\n'.join([_x.strip() for _x in x]))
    df['user'] = df['user'].apply(lambda x: ','.join(map(str, x)))
    df['answers'] = df['response']
    df['ctxs'] = df['knowledge'].apply(lambda x: x[0].split('__knowledge__')[1].strip())
    df['label'] = 1
    df = df[['question', 'ctxs', 'answers', 'label', 'user']]
    dataset = Dataset.from_pandas(df)
    if max_samples is not None:
        indices = list(range(len(dataset)))
        if random_sample:
            random.shuffle(indices)
        dataset = dataset.select(indices[:max_samples])

    return dataset, wow_config
