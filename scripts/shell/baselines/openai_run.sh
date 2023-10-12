python baseline/openai_run.py \
    --data_path ../kcd_data/cached/wow/test_unseen.jsonl --dataset wow \
    --use_kilt_format False \
    --task completion --max_tokens 32 --top_p 0.1 \
    --model_name text-davinci-003 --human_indices generations/wow_human_indices.txt
