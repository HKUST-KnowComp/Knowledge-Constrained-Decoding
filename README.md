# Knowledge Constrained Decoding

Official Code for EMLNP 2023 Paper "KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection".

## Environment

```bash
pip install -r requirements.txt
pip install -e .
```
## Prepare Data

1. First, download WoW dataset through [ParlAI](https://github.com/facebookresearch/ParlAI).
2. Then,

```bash
export WOW_PATH=<PATH to WOW DATASET>
sh scripts/shell/data_process/preprocess_wow.sh 20 $WOW_PATH
```

3. Generate Partial Negative data

```bash
bash scripts/shell/data_process/partial_neg_gen.sh 0 wow 16  # for wow
bash scripts/shell/data_process/partial_neg_gen.sh 0 cnn_dailymail 16  # for cnn/dm data
```

4. Sample Random Negative data (for WoW only)

```bash
bash scripts/shell/data_process/random_neg.sh wow
```

5. Mix the datasets to your liking.

```python

# typo expected
from datasets import load_from_disk

partial_data_path = <CHANGE HERE>
random_data_path = <CHANGE HERE>

partial_data = load_from_disk(partial_data_path)
random_data = load_from_disk(random_data_path)

merged_dataset = concatenate_datasets([partial_data, random_data])
merged_dataset.train_test_split(test_size=0.1)
```
## Train RIPA discrimnator

```bash
sh scripts/shell/train/train_t5_token_classifier.sh 0 EOS 0 0 0 0  # train f
sh scripts/shell/train/train_t5_token_classifier.sh 0 RIPA 0 0 0 1  # finetune RIPA from f
sh scripts/shell/train/train_t5_token_classifier_cnn.sh 0 RIPA 0 0 0 0  # cnn
```

## Run Weighted Decoding

```bash
sh scripts/shell/guided_run.sh 0 fudge RAND wow 8 0 0 0 ''
sh scripts/shell/guided_run.sh 0 nado ALL wow 8 1 0 0 ''
# KWD
sh scripts/shell/guided_run.sh 0 fudge RIPA wow 8 0 0 0 ''
```

## Run MCTS (KCTS)

```bash
sh scripts/shell/ppl_mcts_run.sh 0 RIPA '' wow 8 0 0 0 0 0
```

## Guide GPT 3.5

- Need to train RIPA on GPT2 for this. Checkout `scripts/shell/train/train_token_classifier_gpt.sh`.

```bash
export EXP_ROOT=<ROOT DIRECTORY FOR EXPERIMENT>
sh scripts/shell/openai_guided_run.sh 0 PARTIAL 4 $EXP_ROOT 0 0 3 0 0 0
```

## Evaluation

We use [UniEval](https://arxiv.org/abs/2210.07197) (Zhong et al., 2022) + [MFMA](https://aclanthology.org/2022.findings-naacl.76.pdf) (Lee et al., 2022, for summarization) + Token-based metrics.

```bash
sh scripts/eval/unieval.sh
```

- One can also evaluate the $f$ confidence, using `scripts/eval/class_prob.sh` script.
- Also see `scripts/eval/test_t5_token_classifier.sh` to evaluate the classifier performance.
