ids=$1
type=$2  # [EOS, RIPA, RIPA, ALL, RAND]
bs=$3

export CUDA_VISIBLE_DEVICES=$ids

python scripts/run_openai_ppl_mcts.py \
    --lm_name gpt2-xl \
    --openai_model_name text-davinci-003 \
    --num_labels 2 \
    --attr_idx 1 \
    --dataset wow \
    --test_data_path data/cached/wow/test_unseen.jsonl \
    --output_path generations/ppl_mcts \
    --disc_name text-davinci-003-$type \
    --instruction_model basic \
    --load_checkpoint saved_models/gpt2-xl-DecoderDisc-wow-$type/checkpoint-best/pytorch_model.bin \
    --batch_size $bs \
    --max_num_gen 512 \
    --num_simulations 20 \
    --mock_debug
