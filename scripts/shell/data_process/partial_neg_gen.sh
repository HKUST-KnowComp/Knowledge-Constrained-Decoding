ids=$1
data=$2
bs=$3

# compute number of gpus
arrIDs=(${ids//,/ })
GPU_PER_NODE="${#arrIDs[@]}"

# decide python launcher
launcher="CUDA_VISIBLE_DEVICES=$ids python"

if [ $data = 'wow' ]; then
    data_options="--dataset_name wow --dataset_path data/cached/wow/train.jsonl \
                  --max_neg_samples 10000 --max_new_tokens 64"
elif [ $data = 'cnn_dailymail' ]; then
    data_options="--dataset_path cnn_dailymail --dataset_name cnn_dailymail \
                  --max_neg_samples 100000 --max_new_tokens 64"
else
    echo $data not recognized.
    exit
fi

script="$launcher kcd/partial_negative.py \
    $data_options \
    --per_device_eval_batch_size $bs \
    --temperature 1.4 --top_p 1 --output_dir data/cached \
    --eval_accumulation_steps 200"

eval $script
