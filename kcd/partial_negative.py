from dataclasses import dataclass
from functools import partial
import random

from datasets import concatenate_datasets, Dataset
import torch
from transformers import (HfArgumentParser, Seq2SeqTrainingArguments)
from kcd.classifier_guidance.guided_generation_predictor import GuidedGenerationPredictor

from kcd.instructions import get_instruction
from kcd.token_classifier.dataloader import DataCollatorForSeq2SeqTokenClassification
from kcd.util import load_transformer_LM_tokenizer
from kcd.wizard_of_wikipedia import load_wow
from kcd.summarization import load_summary_data
from kcd.configs import GenerationConfig

@dataclass
class ExperimentArgs:
    dataset_name: str = 'wow'
    dataset_path: str = 'data/wow.jsonl'
    model_name: str = "google/flan-t5-xl"
    instruction_model: str = 'basic'
    max_neg_samples: int = 100000
    load_8bit: bool = True
    first_n: int = 3
    last_n: int = 5
    num_workers: int = 16


def main():
    parser = HfArgumentParser([ExperimentArgs, GenerationConfig, Seq2SeqTrainingArguments])
    args, gen_parameters, train_args = parser.parse_args_into_dataclasses()
    train_args.predict_with_generate = True
    train_args.remove_unused_columns = False  # keep to False

    if args.dataset_name == 'wow':
        data_load_fn = partial(load_wow,
                               max_samples=args.max_neg_samples,
                               random_sample=True)
    elif args.dataset_name in ['cnn_dailymail', 'xsum']:
        data_load_fn = partial(load_summary_data,
                               split='train',
                               max_train_samples=args.max_neg_samples,
                               random_sample=True)
    else:
        raise NotImplementedError

    with train_args.main_process_first(desc="train dataset map pre-processing"):
        dataset, config = data_load_fn(args.dataset_path)
        print(len(dataset))

    load_kwargs = {
        'device_map': 'auto' if args.load_8bit else None,
        'load_in_8bit': args.load_8bit,
        'torch_dtype': torch.float16 if args.load_8bit else torch.bfloat16,
    }
    model, tokenizer = load_transformer_LM_tokenizer(args.model_name, **load_kwargs)
    tokenizer.truncation_side = 'left'
    is_encoder_decoder = model.config.is_encoder_decoder
    if is_encoder_decoder:
        decoder_start_token_id = model.config.decoder_start_token_id

    dataset = dataset.add_column('index', range(len(dataset)))

    def preprocess(example):
        # randomly select knowledge
        # TODO: non-random selection
        non_batch_indices = list(filter(lambda x: x != example['index'], range(len(dataset))))

        answer = example['answers'].strip()
        question = example['question'].strip()
        idx = random.choice(non_batch_indices)
        knowledge = dataset[idx]['ctxs']
        # randomly perturb the answer
        answer_tokens = tokenizer.encode(answer, return_tensors='pt')[0]
        first = args.first_n if args.first_n < len(answer_tokens) else 1
        if len(answer_tokens) - args.last_n > first:
            last = len(answer_tokens) - args.last_n
        else:
            last = len(answer_tokens) - 1
        perturb_idx = random.randint(first, last)
        pert = answer_tokens[:perturb_idx]
        pert_txt = tokenizer.decode(pert, skip_special_tokens=True)

        if 'question' in config['input_columns']:
            input_text = get_instruction(args.instruction_model,
                                         args.dataset_name,
                                         question=question,
                                         knowledge=knowledge)
        else:
            input_text = get_instruction(args.instruction_model,
                                         args.dataset_name,
                                         knowledge=knowledge)

        if is_encoder_decoder:
            tokenized = tokenizer(input_text,
                                  truncation=True,
                                  max_length=tokenizer.model_max_length,
                                  return_tensors='pt')
            tokenized['decoder_input_ids'] = torch.cat(
                [torch.full((1, 1),
                            decoder_start_token_id,
                            dtype=torch.long),
                 pert.unsqueeze(0)],
                dim=1)
        else:
            tokenized = tokenizer(input_text + ' ' + pert_txt,
                                  truncation=True,
                                  max_length=tokenizer.model_max_length,
                                  return_tensors='pt')
            tokenized.pop('token_type_ids', None)  # unused

        tokenized = {k: v[0] for k, v in tokenized.items()}
        tokenized.update(**dict(perturb_idx=perturb_idx, pert_txt=pert_txt))
        return tokenized

    with train_args.main_process_first(desc="train dataset map pre-processing"):
        tokenized_dataset = dataset.map(preprocess,
                                        num_proc=args.num_workers,
                                        remove_columns=dataset.column_names)
    perturb_indices = tokenized_dataset['perturb_idx']
    perturb_prompt = tokenized_dataset['pert_txt']
    tokenized_dataset = tokenized_dataset.remove_columns(['perturb_idx', 'pert_txt'])


    def generate_fn(_model, _tokenizer, inputs):
        generated = _model.generate(**inputs, **gen_parameters.__dict__)
        # get rid of tokens that were already in the input
        if _model.config.is_encoder_decoder:
            generated = generated[:, len(inputs['decoder_input_ids'][0]):]
        else:
            generated = generated[:, len(inputs['input_ids'][0]):]
        return generated

    trainer = GuidedGenerationPredictor(
        generate_fn=generate_fn,
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2SeqTokenClassification(tokenizer),
        tokenizer=tokenizer,
    )
    preds = trainer.predict(tokenized_dataset, **gen_parameters.__dict__)
    preds.predictions[preds.predictions == -100] = tokenizer.pad_token_id
    responses = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

    neg_data = []
    for i, (example, p_txt, p_idx, neg) in enumerate(zip(dataset,
                                                         perturb_prompt,
                                                         perturb_indices,
                                                         responses)):
        target = f'{p_txt} {neg}'.strip()
        label = tokenizer.encode(target, return_tensors='pt')[0]
        label[:p_idx] = 1
        label[p_idx:] = 0
        neg_data.append({
            'answers': target,
            'ctxs': example['ctxs'],  # correct knowledge
            'question': example['question'],
            'label': label.tolist(),
        })
    neg_dataset = Dataset.from_list(neg_data)
    neg_dataset = neg_dataset.filter(lambda x: '\n' in x['question'])

    def _listed_label(example):
        example['label'] = [example['label']]
        return example
    dataset = dataset.map(_listed_label)

    full_data = concatenate_datasets([dataset, neg_dataset])
    full_data.save_to_disk(
        f'data/cached/{args.dataset_name}_train_augmented_neg_{args.model_name.replace("/", "-")}')


if __name__ == '__main__':
    main()
