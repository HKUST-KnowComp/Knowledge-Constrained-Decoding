# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from tqdm.auto import tqdm
from knowledge_source import KnowledgeSource

ks = KnowledgeSource()


def convert_kilt(inputpath, outputpath):
    data = []
    inputdata = open(inputpath, "r")
    for example in tqdm(inputdata):
        d = {}
        ex = json.loads(example)
        d["question"] = ex["claim"]
        d["answers"] = ex["label"]
        d["id"] = ex["id"]

        if ex['label'] == 'NOT ENOUGH INFO':
            d['evidence'] = None
            continue

        evidence_ids = {}
        for ev in ex["evidence"]:
            for ann in ev:
                _, _, wikipedia_title, sentence_id = ann
                if wikipedia_title is None:
                    continue
                if wikipedia_title not in evidence_ids:
                    evidence_ids[wikipedia_title] = {sentence_id}
                else:
                    evidence_ids[wikipedia_title].add(sentence_id)

        evidences = {}
        for title, sent_ids in evidence_ids.items():
            page = ks.get_page_by_id(title)
            if page is None:
                continue
            sentence = []
            sents = [t.strip() + '.' for t in page['text'].split(' . ')]
            for sid in sent_ids:
                try:
                    sentence.append(sents[sid])
                except:
                    pass
            evidences[title] = sentence

        if evidences:
            d["evidence"] = evidences
        else:
            d["evidence"] = None
        data.append(d)
    with open(outputpath, "w") as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    convert_kilt(inputpath, outputpath)
