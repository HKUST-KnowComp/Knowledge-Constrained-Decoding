import os

from datasets import Dataset, load_from_disk
import pandas as pd


def load_wow(path):
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
    df['ctxs'] = df['ctxs'].apply(lambda x: x[0]['text'] if x is not None else None)
    df['answers'] = df['answers'].apply(lambda x: x[0])
    df['label'] = 1
    dataset = Dataset.from_pandas(df)

    return dataset, wow_config


def load_fever(path):
    fever_config = {
        'input_columns': ['ctxs'],
        'instruction': "Evidences:\n{}\n\nGenerate a claim that is"
                       " entirely supported by the evidences above."
    }
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        return dataset, fever_config
    df = pd.read_json(path, lines=True)
    df['answers'] = df['answers'].apply(lambda x: x[0])
    df.loc[df['ctxs'].isna(), 'answers'] = 'NOT ENOUGH INFO'

    def _answer2label(x):
        if x == 'NOT ENOUGH INFO':
            return 0
        elif x == 'SUPPORTS':
            return 1
        elif x == 'REFUTES':
            return 2
        else:
            raise ValueError

    df['label'] = df['answers'].apply(_answer2label)
    # process ctxs
    evidences = []
    for ctx in df['ctxs']:
        if ctx is None:
            evidences.append(None)
            continue
        evid = 0
        evidence = []
        for ev in ctx:
            if ev is None:
                continue
            evidence.append(f'Knowledge {evid}: {ev["text"].strip()}')
            evid += 1
        evidence = '\n'.join(evidence)
        evidences.append(evidence)
    df['ctxs'] = evidences
    dataset = Dataset.from_pandas(df)
    return dataset, fever_config


def main(path, dataset):
    if dataset == 'wow':
        df = load_wow(path)
    elif dataset == 'fever':
        df = load_fever(path)
    else:
        raise ValueError

    print(df)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
