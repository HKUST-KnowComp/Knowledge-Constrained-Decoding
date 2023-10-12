import json
import random
import os

import fire
import pandas as pd


def sample_indices_for_human(task, max_index, size=100):
    """Sample indices for human data."""
    if task == 'wow':
        df = pd.read_json(f'generations/baseline/{task}-openai_gpt-3.5-turbo.jsonl', lines=True)
        good_indices = []
        for i, prompt in enumerate(df['prompt']):
            for msg in prompt:
                if msg['role'] == 'system':
                    knowledge = msg['content'].replace("Use the following knowledge,"
                                                    " but not directly copy, to"
                                                    " generate a concise response: ", "")
            if knowledge.strip() == '' or 'no_passages_used' in knowledge:
                continue
            good_indices.append(i)
    else:
        good_indices = list(range(max_index))
    random.shuffle(good_indices)
    indices = good_indices[:size]
    return indices

def read_generations_from_index_file(generation_file, indices):
    """Read generations from index file."""

    generations = pd.read_json(generation_file, lines=True)
    return generations.iloc[indices]

def main(task='wow', size=100, seed=1234, do_sample=False, do_read=False):
    """Main function."""
    max_indices = {
        'wow': 3924,
        'cnn_dailymail': 1780,
    }
    random.seed(seed)
    if do_sample:
        indices = sample_indices_for_human(task, max_indices[task], size=size)

        with open(f'generations/{task}_human_indices.txt', 'w') as f:
            for idx in indices:
                f.write(str(idx) + '\n')

    target_files = [
        # 'generations/baseline/wow-openai_gpt-3.5-turbo.jsonl',
        # 'generations/fudge/wow-google-flan-t5-xl-fudge-DecoderDisc-wow-RAND.jsonl',
        # 'generations/ppl_mcts/wow-google-flan-t5-xl-DecoderDisc-wow-PARTIAL.jsonl',
        # 'generations/ppl_mcts/wow-google-flan-t5-xl-DecoderDisc-wow-EOS.jsonl',
        # 'generations/baseline/cnn_dailymail-openai_gpt-3.5-turbo.jsonl',
        # 'generations/baseline/cnn_dailymail-google-flan-t5-xl.jsonl',
        # 'generations/ppl_mcts/cnn_dailymail-google-flan-t5-xl-token_f1.jsonl',
        # 'generations/ppl_mcts/cnn_dailymail-google-flan-t5-xl-DecoderDisc-cnn_dailymail-EOS-only_mlp.jsonl',
        'generations/baseline/wow-google-flan-t5-xl.jsonl',
        'generations/ppl_mcts/wow-google-flan-t5-xl-DecoderDisc-wow-EOS-PARTIAL.jsonl',
        'generations/fudge/wow-google-flan-t5-xl-fudge-DecoderDisc-wow-EOS-PARTIAL.jsonl'
    ]

    if do_read:
        indices = []
        with open(f'generations/{task}_human_indices.txt', 'r') as f:
            for line in f:
                idx = int(line.strip())
                indices.append(idx)
        for generation_file in target_files:
            if not os.path.exists(generation_file):
                print(f'{generation_file} does not exist. skipping...')
                continue
            else:
                print(f'{generation_file}')
            generations = read_generations_from_index_file(generation_file, indices)
            out_fname = os.path.splitext(generation_file)[0]
            generations.to_json(f'{out_fname}_human.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    fire.Fire(main)
