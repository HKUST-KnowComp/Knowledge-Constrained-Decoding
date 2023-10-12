N=$1

if [[ $N = '' ]]; then
    N=20
fi

ParlAI=$HOME/ParlAI

# 1. Train
python kcd/wizard_of_wikipedia/preprocess.py \
    --in_file $ParlAI/data/wizard_of_wikipedia/train.json \
    --out_file data/cached/wow/train.jsonl \
    --keep_last_n $N

# 2. valid
python kcd/wizard_of_wikipedia/preprocess.py \
    --in_file $ParlAI/data/wizard_of_wikipedia/valid_random_split.json \
    --out_file data/cached/wow/dev_seen.jsonl \
    --keep_last_n $N
python kcd/wizard_of_wikipedia/preprocess.py \
    --in_file $ParlAI/data/wizard_of_wikipedia/valid_topic_split.json \
    --out_file data/cached/wow/dev_unseen.jsonl \
    --keep_last_n $N

# 3. test
python kcd/wizard_of_wikipedia/preprocess.py \
    --in_file $ParlAI/data/wizard_of_wikipedia/test_random_split.json \
    --out_file data/cached/wow/test_seen.jsonl \
    --keep_last_n $N
python kcd/wizard_of_wikipedia/preprocess.py \
    --in_file $ParlAI/data/wizard_of_wikipedia/test_topic_split.json \
    --out_file data/cached/wow/test_unseen.jsonl \
    --keep_last_n $N
