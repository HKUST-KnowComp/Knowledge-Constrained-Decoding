generation_file=$1

function eval_prob(){
    ids=$1
    task=$2
    name=$3
    skip=$4
    script="CUDA_VISIBLE_DEVICES=$ids python scripts/eval/evaluate_generations_with_classifier.py \
        --model_name google/flan-t5-xl \
        --load_8bit \
        --dataset $task \
        --test_data_path data/cached/wow/test_unseen.jsonl \
        --generations_path $name \
        --load_checkpoint saved_models/flan-t5-xl-DecoderDisc-$task-EOS/checkpoint-best/pytorch_model.bin \
        --batch_size 8"

    if [[ $name == *"alpaca"* ]]; then
        script="$script --causal_lm_generations"
    fi

    if [ $skip ]; then
        script="$script --skip_no_knowledge"
    fi

    eval $script
}

eval_prob 5 wow $generation_file 1
