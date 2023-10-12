import logging
import os
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import fire

from kcd.util import get_logger


def experiment(model,
               tokenizer,
               idx,
               evidence,
               original_claim,
               logger,
               device='cpu',
               prompt_len=3,
               max_new_tokens=40):
    ##################### input prep ###################
    if not isinstance(evidence, list):
        evidence = [evidence]
        original_claim = [original_claim]
        idx = [idx]

    basic_gen_inst = []
    evidence_inst = []
    zero_shot_inst = []
    zero_shot_prompted_inst = []
    prompts = []
    for evid, claim in zip(evidence, original_claim):
        prompt = ' '.join(claim.split(' ')[:prompt_len])

        basic_gen_inst.append(f"Complete the following sentence: {prompt}")
        evidence_inst.append(f"""Complete the following sentence: {evid} {prompt}""")
        zero_shot_inst.append(f"""Generate a claim that is supported by the evidence below.
        evidence: {evid}
        claim:""")
        zero_shot_prompted_inst.append(f"""Generate a claim that is supported by the evidence below.
        evidence: {evid}
        claim: {prompt}""")
        prompts.append(prompt)

    gen_inst_ids = tokenizer(basic_gen_inst, return_tensors='pt', padding=True).to(device)
    evidence_inst_ids = tokenizer(evidence_inst, return_tensors='pt', padding=True).to(device)
    zero_shot_inst_ids = tokenizer(zero_shot_inst, return_tensors='pt', padding=True).to(device)
    zero_shot_prompted_inst_ids = tokenizer(zero_shot_prompted_inst,
                                            return_tensors='pt',
                                            padding=True).to(device)

    all_instructions = {
        'completion': gen_inst_ids,
        '+ evidence': evidence_inst_ids,
        '+ zero-shot-inst': zero_shot_inst_ids,
        '+ zero-shot+prompted': zero_shot_prompted_inst_ids,
    }

    ################ Generation ########################
    def _generate(inputs, top_p=0.95, temperature=0.8, num_beams=8):
        greedy = tokenizer.batch_decode(model.generate(**inputs, max_new_tokens=max_new_tokens),
                                        skip_special_tokens=False)
        topp = tokenizer.batch_decode(model.generate(**inputs,
                                                     max_new_tokens=max_new_tokens,
                                                     do_sample=True,
                                                     top_p=top_p,
                                                     temperature=temperature),
                                      skip_special_tokens=False)
        beam = tokenizer.batch_decode(model.generate(**inputs,
                                                     num_beams=num_beams,
                                                     max_new_tokens=max_new_tokens),
                                      skip_special_tokens=False)

        return greedy, topp, beam

    completions = {}
    for key, inst in all_instructions.items():
        completions[key] = _generate(inst)

    data = []
    for i in range(len(idx)):
        _data = {
            'data_idx': idx[i],
            'evidence': evidence[i],
            'prompt': prompts[i],
            'original_claim': original_claim[i],
            'results': {},
        }

        logger.info('#' * 40 + 'fever test set index %d ' + '#' * 40, idx[i])
        logger.info('evidence: %s', evidence[i])
        logger.info('prompt: %s', prompts[i])
        logger.info('original claim: %s', original_claim[i])

        for key, result in completions.items():
            _data['results'][key] = {}
            _data['results'][key]['greedy'] = result[0][i]
            _data['results'][key]['top_p'] = result[1][i]
            _data['results'][key]['beam'] = result[2][i]

            logger.info(
                """[%s]
                        \t[greedy]
                        \t%s
                        \t[top_p=0.95, temp=0.8]
                        \t%s
                        \t[beam=8]
                        \t%s""", key, result[0][i], result[1][i], result[2][i])

        data.append(_data)
    return data


def main(
    pretrained_model='google/flan-t5-xl',
    fever_data_path='data/fever/paper_test.jsonl',
    outfname='outputs/fever_prompt_baseline.jsonl',
    prompt_len=1,
    max_new_tokens=20,
    batch_size=16,
    end_idx=1000,
):
    ############## logging ####################
    logging.basicConfig(level=logging.INFO)
    logger = get_logger('logs/prompt_baseline_test.log')
    ###########################################

    wiki_added_fever_path = os.path.splitext(fever_data_path)[0] + '+wiki.jsonl'
    fever = pd.read_json(wiki_added_fever_path, lines=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #################### MODEL LOADING ########
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if 't5' in pretrained_model or 't0' in pretrained_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model,
                                                      device_map='balanced_low_0',
                                                      load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        # open-ended generation
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model.config.bos_token_id = model.config.eos_token_id
    # model = model.to(device)
    model.eval()
    print('model loading finished')
    ############################################
    outfile = open(outfname, 'w')
    indices = []
    evidences = []
    claims = []
    for idx, df in fever.iterrows():
        if len(df['wiki_extracted']) == 0:
            continue
        evidence = df['wiki_extracted'][0]  # first evidence
        evidence = evidence.replace('-LRB-', '(')
        evidence = evidence.replace('-RRB-', ')')
        claim = df['claim']

        indices.append(idx)
        evidences.append(evidence)
        claims.append(claim)

        if (idx + 1) % batch_size == 0:
            batch_result = experiment(model,
                                      tokenizer,
                                      indices,
                                      evidences,
                                      claims,
                                      logger,
                                      device=device,
                                      prompt_len=prompt_len,
                                      max_new_tokens=max_new_tokens)
            for res in batch_result:
                json.dump(res, outfile)
                outfile.write('\n')
            indices = []
            evidences = []
            claims = []
        if idx > end_idx:
            break
    outfile.close()


if __name__ == '__main__':
    fire.Fire(main)
