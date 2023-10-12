# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from tqdm.auto import tqdm
from knowledge_source import KnowledgeSource

# get the knowledge souce
ks = KnowledgeSource()


def convert_kilt(inputpath, outputpath):
    data = []
    inputdata = open(inputpath, "r")
    for example in tqdm(inputdata):
        d = {}
        ex = json.loads(example)
        d["question"] = ex["input"]
        answers = set()
        for a in ex["output"]:
            if "answer" in a:
                answers.add(a["answer"])
        d["answers"] = list(answers)
        d["id"] = ex["id"]
        passages = []

        if 'provenance' not in ex['output'][0]:
            d["ctxs"] = None
            data.append(d)
            continue

        for c in ex["output"][0]["provenance"]:
            page = ks.get_page_by_id(c["wikipedia_id"])
            text = []
            if c['start_paragraph_id'] == c['end_paragraph_id']:
                # single paragraph
                pid = c['start_paragraph_id']
                text = page['text'][pid][c['start_character']:c['end_character'] + 1]
            else:
                for pid in range(c['start_paragraph_id'], c['end_paragraph_id'] + 1):
                    if pid == c['start_paragraph_id']:  # start
                        t = page['text'][pid][c['start_character']:]
                    elif pid == c['end_paragraph_id']:  # end
                        t = page['text'][pid][:c['end_character'] + 1]
                    else:  # inbetween
                        t = page['text'][pid]
                    text.append(t)

            p = {
                "text": text,
                "title": page["wikipedia_title"],
                "wikipedia_id": page["wikipedia_id"]
            }
            passages.append(p)
        d["ctxs"] = passages
        data.append(d)
    with open(outputpath, "w") as fout:
        for entry in data:
            json.dump(entry, fout)
            fout.write('\n')


if __name__ == "__main__":
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    convert_kilt(inputpath, outputpath)
