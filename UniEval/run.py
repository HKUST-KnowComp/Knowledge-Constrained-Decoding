from dataclasses import dataclass
import json
import os

from datasets import Dataset, load_dataset
import editdistance
import pandas as pd
from transformers import HfArgumentParser, AutoTokenizer
import evaluate

from UniEval.utils import convert_to_json
from UniEval.evaluator import get_evaluator
from kcd.evaluation.token_f1_score import TokenF1Score

MAIN_MODEL = 'google/flan-t5-xl'


def load_summary_data(path: str, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL)
    if path == 'cnn_dailymail':
        dataset = load_dataset(path, '3.0.0')['test']
        dataset = dataset.rename_column('article', 'ctxs')
        dataset = dataset.rename_column('highlights', 'answers')
    elif path == 'xsum':
        dataset = load_dataset(path)['test']
        dataset = dataset.rename_column('document', 'ctxs')
        dataset = dataset.rename_column('summary', 'answers')
    else:
        raise ValueError(f'Unknown dataset: {path}')

    # Tokenize the dataset and filter out samples that are too long
    def get_doc_len(examples):
        return {'doc_len': len(tokenizer.encode(examples['ctxs']))}

    dataset = dataset.map(get_doc_len)
    # -25 for instructions
    dataset = dataset.filter(lambda x: x['doc_len'] <= tokenizer.model_max_length - 25)

    return dataset


def load_dialog_data(path: str,
                     dialog_separator=' \n ',
                     dialog_eos='\n\n',
                     dialog_knowledge_eos='\n'):
    df = pd.read_json(path, lines=True)
    df['question'] = df['history'].apply(
        lambda x: dialog_separator.join([_x.strip() for _x in x]) + dialog_eos)
    df['answers'] = df['response']
    df['ctxs'] = df['knowledge'].apply(
        lambda x: x[0].split('__knowledge__')[1].strip() + dialog_knowledge_eos)
    df = df[['question', 'ctxs', 'answers']]
    dataset = Dataset.from_pandas(df)
    return dataset


def load_generations(path: str, causal_lm_generations=False, indices=None):
    df = pd.read_json(path, lines=True)
    if indices is not None:
        df = df.iloc[indices]
        df.reset_index(inplace=True)
    if isinstance(df['response'][0], dict):
        # for chat GPT
        generations = []
        for response in df['response']:
            if 'text' in response['choices'][0]:
                gen = response['choices'][0]['text'].strip()
            else:
                gen = response['choices'][0]['message']['content']
            generations.append(gen)
        return generations, df['index'].tolist()

    if causal_lm_generations:
        generations = []
        for i, response in enumerate(df['response']):
            try:
                gen = response.split('### Response:')[1]
                if not gen.strip():
                    print(f"No response found in generation. index: {i}")
                    gen = "no response"
            except:
                print(f"No response found in generation. index: {i}")
                gen = "no response"
            generations.append(gen)
        return generations, df['index'].tolist()

    return df['response'].tolist(), df['index'].tolist()


@dataclass
class Args:
    task: str = 'summarization'  # choices=['summarization', 'dialogue', 'fact']
    generations_path: str = 'generations.jsonl'
    dataset_path: str = 'cnn_dailymail'
    causal_lm_generations: bool = False
    automatic_eval: bool = True
    skip_unieval: bool = False  # skip UniEval evaluation
    save_name: str = None
    skip_no_knowledge: bool = False
    human_indices: str = None


def main():
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]

    human_indices = None
    if args.human_indices:
        with open(args.human_indices, 'r') as f:
            human_indices = [int(l.strip()) for l in f]

    # load model generation
    output_list, indices = load_generations(args.generations_path, args.causal_lm_generations,
                                            human_indices)

    # load source data
    if args.task == 'summarization':
        raw_data = load_summary_data(args.dataset_path)
        if len(raw_data) != len(output_list):
            raw_data = raw_data.select(indices)
        src_list = raw_data['ctxs']
        ref_list = raw_data['answers']
        ref_list = [x.replace('\n', ' ') for x in ref_list]
        data = convert_to_json(src_list=src_list, ref_list=ref_list, output_list=output_list)
    elif args.task == 'dialogue':
        """
        #################### dialog format References #######################
        # a list of dialogue histories
        src_list = ['hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n']
        # a list of additional context that should be included into the generated response
        context_list = ['the 3 horizontal line menu on apps and websites is called a hamburger button .\n']
        # a list of model outputs to be evaluated
        output_list = ['i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?']
        """
        dialog_kwargs = dict(
            dialog_separator=' \n ',
            dialog_eos='\n\n',
            dialog_knowledge_eos='\n',
        )
        raw_data = load_dialog_data(args.dataset_path, **dialog_kwargs)
        if len(raw_data) != len(output_list):
            raw_data = raw_data.select(indices)
        src_list = raw_data['question']
        context_list = raw_data['ctxs']
        ref_list = raw_data['answers']
        if args.skip_no_knowledge:
            print('skipping no knowledge')
            filtered_src, filtered_ctx, filtered_output, filtered_ref = [], [], [], []
            for src, ctx, out, ref in zip(src_list, context_list, output_list, ref_list):
                if ctx.strip() == 'no_passages_used':
                    continue
                filtered_src.append(src)
                filtered_ctx.append(ctx)
                filtered_output.append(out)
                filtered_ref.append(ref)
            src_list, context_list, output_list, ref_list = filtered_src, filtered_ctx, filtered_output, filtered_ref
        data = convert_to_json(src_list=src_list,
                               context_list=context_list,
                               output_list=output_list)
    elif args.task == 'fact':
        # kwarg_list = ['src_list', 'output_list']
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown task: {args.task}')

    if not args.skip_unieval:
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(args.task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, overall=False, print_result=True)
        # eval_scores: list[dict]
        # format eval_scores from list[dict] into dict[list]
        eval_scores = {k: [d[k] for d in eval_scores] for k in eval_scores[0]}
        mean_eval_scores = {k: sum(v) / len(v) for k, v in eval_scores.items()}
    else:
        print("Skipped UniEval evaluation.")
        mean_eval_scores = {}

    if args.automatic_eval:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load('rouge')
        chrf = evaluate.load("chrf")
        meteor = evaluate.load('meteor')

        metrics = {'P': [], 'R': [], 'F1': [], 'KP': [], 'KR': [], 'KF1': [], 'K-Copy': []}

        knowledge_list = src_list if args.task == 'summarization' else context_list
        for guess, answer, knowledge in zip(output_list, ref_list, knowledge_list):
            p, r, f1 = TokenF1Score.compute(guess, [answer], expose_p_and_r=True)
            kp, kr, kf1 = TokenF1Score.compute(guess, [knowledge], expose_p_and_r=True)
            lev_dist = editdistance.eval(guess, knowledge)
            knowledge_copy = 1 - lev_dist / max(len(knowledge), len(guess))
            metrics['P'].append(p)
            metrics['R'].append(r)
            metrics['F1'].append(f1)
            metrics['KP'].append(kp)
            metrics['KR'].append(kr)
            metrics['KF1'].append(kf1)
            metrics['K-Copy'].append(knowledge_copy)

        # average
        for key in metrics.keys():
            metrics[key] = sum(metrics[key]) / len(metrics[key])

        metrics['BLEU'] = bleu.compute(predictions=output_list,
                                       references=[[x] for x in ref_list])['bleu']
        metrics['K-BLEU'] = bleu.compute(predictions=output_list,
                                         references=[[x] for x in knowledge_list])['bleu']
        metrics['RougeL'] = rouge.compute(predictions=output_list, references=ref_list)['rougeL']
        metrics['K-RougeL'] = rouge.compute(predictions=output_list, references=ref_list)['rougeL']

        metrics['chrF'] = chrf.compute(predictions=output_list, references=ref_list)['score']
        metrics['meteor'] = meteor.compute(predictions=output_list, references=ref_list)['meteor']

        for key, value in metrics.items():
            mean_eval_scores[key] = value
    else:
        print("Skipped automatic evaluation.")

    # save eval_scores
    if not args.save_name:
        name = os.path.splitext(os.path.basename(args.generations_path))[0]
    else:
        name = args.save_name
    if args.human_indices:
        name += '_human'
    if args.skip_no_knowledge:
        outfname = f'eval_results/skip_no_knowledge/unieval_{name}.json'
    else:
        outfname = f'eval_results/unieval_{name}.json'
    if os.path.isfile(outfname):
        with open(outfname) as f:
            prev_eval_scores = json.load(f)
        prev_eval_scores.update(mean_eval_scores)
        mean_eval_scores = prev_eval_scores

    with open(outfname, 'w') as f:
        json.dump(mean_eval_scores, f, indent=2)


if __name__ == '__main__':
    main()
