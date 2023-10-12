import pandas as pd
import datasets
import random

from termcolor import colored
from transformers import AutoTokenizer
import fire

def get_neg_data_only(dataset):
    for i, l in enumerate(dataset['label']):
        if len(l) > 1:
            break
    return dataset[i + 1:]

def detokenize(tokens):
    return ''.join([' ' if tok == '▁' else tok.replace('▁', ' ') for tok in tokens])


def analyze(neg_dataset, idx, task, tokenizer, full_data, verbose=False):
    def verbose_print(*args):
        if verbose:
            print(*args)
    if task == 'wow':
        verbose_print('history\n\n', neg_dataset['question'][idx])
        verbose_print('\n\nknowledge\n\n', neg_dataset['ctxs'][idx])

        last_idx = sum(neg_dataset['label'][idx]) - 1
        tokenized = tokenizer.tokenize(neg_dataset['answers'][idx])
        original_tokens = detokenize(tokenized[:last_idx]).strip()
        hallucinated_tokens = detokenize(tokenized[last_idx:]).strip()

        verbose_print("\n\nanswer\n\n", original_tokens + colored(hallucinated_tokens, 'red'))

        verbose_print('\n\noriginal answer\n\n', full_data[full_data['history'] == neg_dataset['question'][idx]]['response'].tolist()[0])
    elif task == 'cnn':
        verbose_print('document\n\n', neg_dataset['ctxs'][idx])

        last_idx = sum(neg_dataset['label'][idx]) - 1
        tokenized = tokenizer.tokenize(neg_dataset['answers'][idx])
        original_tokens = detokenize(tokenized[:last_idx])
        hallucinated_tokens = detokenize(tokenized[last_idx:])

        verbose_print("\n\nsummary\n\n", original_tokens + colored(hallucinated_tokens, 'red'))

        for i, art in enumerate(full_data['article']):
            if art == neg_dataset['ctxs'][idx]:
                verbose_print('\n\noriginal summary\n\n', full_data[i]['highlights'])
                break
    else:
        raise ValueError

    return hallucinated_tokens


def main(task: str='wow', verbose: bool=False):
    """
    task: wow or cnn
    """
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')

    if task == 'wow':
        wow_df = pd.read_json('data/cached/wow/train.jsonl', lines=True)
        wow_df['history'] = wow_df['history'].apply(lambda x: '\n'.join(x))
        partial_data = datasets.load_from_disk('data/cached/wow_train_augmented_neg_google-flan-t5-xl')
        partial_negative = get_neg_data_only(partial_data)
        full_data = wow_df
    elif task == 'cnn':
        cnn_data = datasets.load_dataset('cnn_dailymail', '3.0.0')['train']
        partial_data = datasets.load_from_disk('data/cached/cnn_dailymail_train_augmented_neg_google-flan-t5-xl')
        partial_negative = get_neg_data_only(partial_data)
        full_data = cnn_data
    else:
        raise ValueError

    for i in random.sample(range(len(partial_negative['label'])), 10):
        analyze(partial_negative, i, task, tokenizer, full_data, verbose=verbose)

if __name__ == '__main__':
    fire.Fire(main)
