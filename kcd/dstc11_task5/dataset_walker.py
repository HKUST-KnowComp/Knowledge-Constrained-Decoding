import os
import json

from datasets import Dataset

from .knowledge_reader import KnowledgeReader


def load_dstc_data(data_path):
    data = DatasetWalker(data_path, labels=True, incl_knowledge=True, do_format=True)
    dataset = Dataset.from_list([x for x in data])
    dataset = dataset.filter(lambda x: x['answers'] is not None)
    config = {
        'input_columns': ['ctxs', 'question'],
        'instruction': "### History:\n{}\n\n### Knowledge:\n{}"
                       "\n\nGiven the dialog history and a relevant knowledge above,"
                       " generate a knowledgeable, usefule, and helpful answer."
    }
    return dataset, config


class DatasetWalker:
    """
    Copied from https://github.com/alexa/dstc11-track5/blob/main/scripts/dataset_walker.py
    Adjusted by Sehyun Choi, 2023
    """
    EOT = '</eot>'

    def __init__(self,
                 data_path,
                 labels=True,
                 labels_file=None,
                 incl_knowledge=True,
                 do_format=True):
        dataset = os.path.basename(data_path)
        dataroot = os.path.dirname(data_path)

        path = dataroot

        if dataset not in ['train', 'val', 'test']:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        logs_file = os.path.join(path, dataset, 'logs.json')
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if labels is True:
            if labels_file is None:
                labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

        self._incl_knowledge = incl_knowledge
        if self._incl_knowledge is True:
            self._knowledge = KnowledgeReader(dataroot)

        self.do_format = do_format

    def __getitem__(self, idx):
        log = self.logs[idx]
        if self.labels is not None:
            label = self.labels[idx]
            if self._incl_knowledge is True and label['target'] is True:
                for idx, snippet in enumerate(label['knowledge']):
                    domain = snippet['domain']
                    entity_id = snippet['entity_id']
                    doc_type = snippet['doc_type']
                    doc_id = snippet['doc_id']

                    if doc_type == 'review':
                        sent_id = snippet['sent_id']
                        sent = self._knowledge.get_review_sent(domain, entity_id, doc_id, sent_id)
                        label['knowledge'][idx]['sent'] = sent

                    elif doc_type == 'faq':
                        doc = self._knowledge.get_faq_doc(domain, entity_id, doc_id)
                        question = doc['question']
                        answer = doc['answer']

                        label['knowledge'][idx]['question'] = question
                        label['knowledge'][idx]['answer'] = answer
        else:
            label = None

        if self.do_format:
            return self.format_log(log, label=label)
        return log, label

    def format_log(self, log, label=None):
        data = {}
        history = []
        speakers = []
        for turn in log:
            speaker = 'User' if turn['speaker'] == 'U' else 'System'
            turn_text = f"{speaker}: {turn['text']}"
            history.append(turn_text)
            speakers.append('0' if speaker == 'User' else '1')
        data['question'] = self.EOT.join(history)
        data['user'] = ','.join(speakers)

        if label is None or not label['target']:
            data['ctxs'] = None
            data['answers'] = None
        else:
            knowledges = []
            for knowledge in label['knowledge']:
                if 'sent' in knowledge:
                    knowledges.append(knowledge['sent'])
                else:
                    knowledges.append(f"Q: {knowledge['question']}\nA: {knowledge['answer']}")
            data['ctxs'] = '\n'.join(knowledges)
            data['answers'] = label['response']
            data['label'] = 1

        return data

    def __len__(self):
        return len(self.logs)
