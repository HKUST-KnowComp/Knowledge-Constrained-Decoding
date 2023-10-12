from dataclasses import dataclass
import json
import os
import time
from termcolor import colored

from transformers import HfArgumentParser
from tqdm.auto import tqdm

from kcd.openai_module import OpenAIModel, OpenAIAPIParameters
from kcd.text_data import load_text_data


@dataclass
class ExperimentArgs:
    data_path: str = 'data/wow-dev-kilt-processed.jsonl'
    output_path: str = 'generations/baseline'
    model_name: str = "gpt-3.5-turbo"
    dataset: str = 'wow'
    use_kilt_format: bool = True
    task: str = 'chat'
    continue_from: int = 0
    debug: bool = False
    skip_no_knowledge: bool = False
    instruction_model: str = 'basic'
    human_indices: str = None


def main():
    parser = HfArgumentParser((ExperimentArgs, OpenAIAPIParameters))
    args, parameters = parser.parse_args_into_dataclasses()

    text_dataset = load_text_data(path=args.data_path,
                                  task=args.task,
                                  instruction_model=args.instruction_model,
                                  use_kilt_format=args.use_kilt_format)
    if args.human_indices:
        with open(args.human_indices) as f:
            indices = [int(i) for i in f.readlines()]
        text_dataset = text_dataset.select(indices)
    model = OpenAIModel(model_name=args.model_name, task=args.task)

    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = args.dataset
    if args.use_kilt_format:
        dataset_name += '-kilt'
    out_fname = os.path.join(args.output_path, f'{dataset_name}-openai_{args.model_name}')
    if args.human_indices:
        out_fname += '_human'
    out_fname += '.jsonl'
    fout = open(out_fname, 'a')
    for i, example in tqdm(enumerate(text_dataset), total=len(text_dataset)):
        if i < args.continue_from:
            continue
        if args.debug:
            full_response = {'dummy': 'dummy'}
            completion = 'dummy'
        elif args.skip_no_knowledge and "no_passages_used" in example['prompt']:
            full_response = {'choices': [{'text': 'skipped'}]}
            completion = 'skipped due to no knowledge'
        else:
            completion = None
            try_count = 0
            start = time.time()
            while completion is None:
                if try_count > 5:
                    print(
                        f"Stop trying after {try_count} tries and {time.time() - start:.2f} seconds."
                    )
                    print(f"You can resume by setting --continue_from={i}")
                    exit(1)
                try:
                    try_count += 1
                    completion, full_response = model(example['prompt'], parameters)
                except:
                    print("OpenAI Rate Limit reached. Sleeping for 5 minutes.")
                    time.sleep(300)
            if try_count > 0:
                print(
                    f"exited while loop after {try_count} tries and {time.time() - start:.2f} seconds"
                )
        result = {
            'prompt': example['prompt'],
            'response': full_response,
            'gold': example['completion'],
            'index': i,
        }
        fout.write(json.dumps(result) + '\n')
        print("prompt:\n")
        if args.task == 'chat':
            for msg in example["prompt"]:
                if msg['role'] == 'user':
                    print(colored('User: ' + msg['content'], 'green'))
                elif msg['role'] == 'assistant':
                    print(colored('Assistant: ' + msg['content'], 'blue'))
                else:
                    print(colored('System: ' + msg['content'], 'red'))
        else:
            print(example["prompt"])
        print()
        print(f"{args.model_name} Response:\n")
        print(completion + '\n\n')
        print(f"Gold Response:\n")
        print(example["completion"] + '\n\n')
    fout.close()


if __name__ == "__main__":
    main()
