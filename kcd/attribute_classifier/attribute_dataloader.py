from statistics import mean, median
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import BatchEncoding


def get_attribute_dataloader(dataname_or_data,
                             tokenizer,
                             max_length: int = 256,
                             batch_size: int = 32,
                             split: str = 'test',
                             num_workers: int = 0):
    dataset = AttributeDataset(dataname_or_data, tokenizer, max_length=max_length, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == 'train',
        num_workers=num_workers,
    )


class AttributeDataset(Dataset):

    def __init__(self,
                 dataname_or_data: str,
                 tokenizer,
                 max_length: int = 256,
                 split: str = 'test',
                 show_stats: bool = False) -> None:
        if isinstance(dataname_or_data, str):
            data = load_dataset(dataname_or_data)
        else:
            data = dataname_or_data
        data = data[split]

        self.labels = data['label']
        self.texts = tokenizer(data['sentence'],
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=max_length)

        if show_stats:
            print(f'[split]: {split}')
            lengths = [len(tokenizer.tokenize(x)) for x in data['sentence']]
            print('text length stats:')
            print(
                f'max: {max(lengths)}, mean: {mean(lengths)}, min: {min(lengths)}, median: {median(lengths)}'
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.texts.items()}
        data = BatchEncoding(data, tensor_type='pt')
        label = self.labels[idx]

        data['labels'] = label

        return data