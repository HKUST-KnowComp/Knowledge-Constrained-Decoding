from dataclasses import dataclass, field
from functools import partial
from typing import Union, Optional, List

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin

from kcd.instructions import get_instruction, WOW_CLASSFICATION_INSTRUCTION
from kcd.kilt.load_kilt_data import load_fever
from kcd.dstc11_task5 import load_dstc_data
from kcd.summarization import load_summary_data
from kcd.util import shift_right

@dataclass
class DataCollatorForSeq2SeqTokenClassification(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    other_features_to_pad: List[str] = field(default_factory=list)
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features
                 ] if label_name in features[0].keys() else None
        other_features = {
            k : [feature[k] for feature in features] for k in  self.other_features_to_pad
        }
        decoder_input_ids = [feature['decoder_input_ids'] for feature in features]

        no_labels_features = [{
            k: v for k, v in feature.items()
            if k not in (label_name, 'decoder_input_ids', *self.other_features_to_pad)
        } for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = max([len(ids) for ids in decoder_input_ids])
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        def pad_tensor(tensor, pad_id, seqlen):
            if padding_side == "right":
                return [to_list(x) + [pad_id] * (seqlen - len(x)) for x in tensor]
            return [[pad_id] * (seqlen - len(x)) + to_list(x) for x in tensor]

        batch['decoder_input_ids'] = pad_tensor(decoder_input_ids, self.tokenizer.pad_token_id, sequence_length)
        batch['decoder_input_ids'] = torch.tensor(batch['decoder_input_ids'], dtype=torch.int64)

        if labels is None:
            return batch
        batch[label_name] = pad_tensor(labels, self.label_pad_token_id, sequence_length)
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)

        if not other_features:
            return batch

        for k, v in other_features.items():
            seqlen = max([len(ids) for ids in v])
            padded = pad_tensor(v, self.tokenizer.pad_token_id, seqlen)
            batch[k] = torch.tensor(padded, dtype=torch.int64)

        return batch


def load_data(args,
              tokenizer,
              is_encoder_decoder=False,
              decoder_start_token_id=0,
              instruction_model='basic',
              zeroshot_classification=False,
              get_knowledge_ids=False):
    if args.dataset == 'fever':
        load_fn = load_fever
    elif args.dataset == 'wow':
        if args.use_kilt_format:
            from kcd.kilt.load_kilt_data import load_wow
        else:
            from kcd.wizard_of_wikipedia import load_wow
        load_fn = load_wow
    elif args.dataset == 'dstc11_task5':
        load_fn = load_dstc_data
    elif args.dataset in ('cnn_dailymail', 'xsum'):
        load_fn = load_summary_data
    else:
        raise ValueError

    def _tokenize(config: dict, example):
        # ['question', 'answers', 'id', 'ctxs', 'label']
        knowledge = example['ctxs'].strip()
        question = example['question'].strip()
        answer = example['answers'].strip()
        if zeroshot_classification:
            classi_text = WOW_CLASSFICATION_INSTRUCTION.format(question, knowledge, answer)
            classi_inputs = tokenizer(classi_text,
                                      return_tensors='pt',
                                      max_length=tokenizer.model_max_length,
                                      return_token_type_ids=False,
                                      truncation=True)
            classi_inputs['labels'] = torch.LongTensor([example['label']])
            classi_inputs = {k: v[0] for k, v in classi_inputs.items()}
            return classi_inputs

        if 'question' in config['input_columns']:
            input_text = get_instruction(instruction_model,
                                         args.dataset,
                                         question=question,
                                         knowledge=knowledge)
            target = answer
        else:
            input_text = get_instruction(instruction_model, args.dataset, knowledge=knowledge)
            target = answer

        if is_encoder_decoder:
            tokenized = tokenizer(input_text,
                                  text_target=target,
                                  max_length=tokenizer.model_max_length,
                                  return_tensors='pt',
                                  truncation=True)
            if not hasattr(args, 'sft') or not args.sft:
                # TODO: need to add decoder start token...
                tokenized['decoder_input_ids'] = shift_right(tokenized['labels'],
                                                             decoder_start_token_id)
                if isinstance(example['label'], list):
                    if len(example['label']) == 1:
                        tokenized['labels'] = torch.full_like(tokenized['decoder_input_ids'], example['label'][0])
                    elif hasattr(args, 'sequence_label') and args.sequence_label:
                        tokenized['labels'] = torch.full_like(tokenized['decoder_input_ids'], example['label'][-1])
                    else:
                        label = torch.LongTensor([example['label']])
                        if not tokenized['decoder_input_ids'].shape[1] == label.shape[1]:
                            # The indexing of label is wrong because of .strip()
                            label = label[:, :-1]
                            assert tokenized['decoder_input_ids'].shape[1] == label.shape[1]
                        tokenized['labels'] = shift_right(label, 1)
                else:
                    tokenized['labels'] = torch.full_like(tokenized['decoder_input_ids'], example['label'])
        else:
            target_len = len(tokenizer.encode(target))
            tokenized = tokenizer(f"{input_text}\n\n{target}",
                                  max_length=tokenizer.model_max_length,
                                  return_tensors='pt',
                                  truncation=True)
            if hasattr(args, 'sft') and args.sft:
                label = tokenized['input_ids'].clone()
                label[:, :-target_len] = -100
                tokenized['labels'] = label
            else:
                full_seqlen = tokenized['attention_mask'].sum(-1)
                if isinstance(example['label'], list):
                    if len(example['label']) == 1:
                        label = torch.full((target_len,), example['label'][0], dtype=int).tolist()
                    elif hasattr(args, 'sequence_label') and args.sequence_label:
                       label = torch.full((target_len,), example['label'][-1], dtype=int).tolist()
                    else:
                        label = example['label']
                        if not target_len == len(label):
                            # The indexing of label is wrong because of .strip()
                            label = label[:-1]
                else:
                    label = torch.full((target_len,), example['label'], dtype=int).tolist()
                label = [[-100] * (full_seqlen - len(label)) + label]
                tokenized['labels'] = label
        if get_knowledge_ids:
            tokenized['knowledge_ids'] = tokenizer(knowledge, return_tensors='pt').input_ids
        tokenized = {k: v[0] for k, v in tokenized.items()}

        return tokenized

    datasets = {}
    for split, path in zip(['train', 'validation', 'test'],
                           [args.train_data_path, args.validation_data_path, args.test_data_path]):
        if not path:
            datasets[split] = None
            continue
        dataset, config = load_fn(path)
        filtered = dataset.filter(lambda x: x['ctxs'] is not None)
        mapped = filtered.map(partial(_tokenize, config), remove_columns=dataset.column_names)
        datasets[split] = mapped
    return datasets
