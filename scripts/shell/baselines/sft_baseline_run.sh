ids=$1
model=$2
task=$3  # [wow | summarization]
# compute number of gpus
arrIDs=(${ids//,/ })
GPU_PER_NODE="${#arrIDs[@]}"

# decide python launcher
if [ $GPU_PER_NODE = 1 ]; then
    echo "Using 1 GPU: use simple python launcher..."
    launcher="CUDA_VISIBLE_DEVICES=$ids python"
else
    echo "Using multi-GPU: using torchrun launcher..."
    launcher="CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE"
fi

if [ $task = 'wow' ]; then
    task_options="--data_path data/cached/wow/test_unseen.jsonl --dataset wow --max_new_tokens 32"
elif [ $task = 'summarization' ]; then
    task_options="--data_path cnn_dailymail --dataset cnn_dailymail --max_new_tokens 64"
else
    echo $task not defined.
    exit
fi

script="$launcher baseline/huggingface_run.py \
    $task_options \
    --use_kilt_format False \
    --task completion --top_p 0.95 \
    --model_name $model \
    --load_8bit \
    --load_checkpoint saved_models/flan-t5-xl-sft-wow/checkpoint-best/pytorch_model.bin \
    --per_device_eval_batch_size 4 \
    --output_dir generations/baseline \
    --predict_with_generate"

eval $script
