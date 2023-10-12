from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, DataCollatorWithPadding
from sklearn.metrics import classification_report

from kcd.token_classifier.dataloader import load_data
from kcd.util import load_transformer_LM_tokenizer

@dataclass
class ExperimentArgs:
    model_name: str = field(default="google/flan-t5-xl")
    load_8bit: bool = True
    instruction_model: str = 'basic'  # 'basic' or 'alpaca'
    dataset: str = 'wow'  # 'wow' or 'fever'
    use_kilt_format: bool = False
    test_data_path: str = "data/cached/wow/test_unseen.jsonl"
    batch_size: int = 1


def main():
    parser = HfArgumentParser([ExperimentArgs])
    args = parser.parse_args_into_dataclasses()[0]
    args.train_data_path = None
    args.validation_data_path = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_kwargs = {
        'device_map': 'auto' if args.load_8bit else None,
        'load_in_8bit': args.load_8bit,
        'torch_dtype': torch.float16 if args.load_8bit else torch.bfloat16,
    }
    model, tokenizer = load_transformer_LM_tokenizer(args.model_name, **load_kwargs)
    model.eval()
    tokenizer.truncation_side = 'left'

    dataset = load_data(args,
                        tokenizer,
                        zeroshot_classification=True,
                        is_encoder_decoder=model.config.is_encoder_decoder,
                        instruction_model=args.instruction_model)['test']
    collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=collator,
                            shuffle=False)

    pos_idx = tokenizer.encode('Yes')[0]
    neg_idx = tokenizer.encode('No')[0]

    preds = []
    labels = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(device)
        label = batch.pop('labels')
        if model.config.is_encoder_decoder:
            batch['decoder_input_ids'] = torch.full((batch['input_ids'].shape[0], 1),
                                                     model.config.decoder_start_token_id,
                                                     dtype=torch.long,
                                                     device=device)
        with torch.inference_mode():
            logits = model(**batch).logits
            probs = torch.softmax(logits[:, -1, [neg_idx, pos_idx]], dim=-1)
            pred = torch.argmax(probs, dim=-1)

        preds.append(pred.cpu())
        labels.append(label.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    print(classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    main()
