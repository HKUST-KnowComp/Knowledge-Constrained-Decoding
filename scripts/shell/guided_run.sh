ids=$1
exp=$2
metric=$3
task=$4
bs=$5
V2=$6
human=$7
quick=$8
cont=$9
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

DATADIR=data/cached
OUTDIR=generations
CKPTDIR=saved_models
CKPTNAME=best

if [[ $bs = '' ]]; then
    bs=16
fi

model_name=flan-t5-xl-DecoderDisc-$task-$metric
disc_name=DecoderDisc-$task-$metric

if [ $V2 = 1 ]; then
    model_name=$model_name-v2
    disc_name=$disc_name-v2
fi

if [ $task = 'wow' ]; then
    task_options="--test_data_path $DATADIR/wow/test_unseen.jsonl --dataset wow --max_new_tokens 32"
elif [ $task = 'cnn_dailymail' ]; then
    task_options="--test_data_path cnn_dailymail --dataset cnn_dailymail --max_new_tokens 64"
else
    echo $task not defined.
    exit
fi

elif [[ $exp = 'fudge' ]]; then

script="$launcher scripts/run_guided_generation.py \
    $task_options \
    --use_kilt_format False \
    --top_p 0.95 --top_k 50 --temperature 1.0 \
    --model_name google/flan-t5-xl \
    --guidance_method fudge \
    --load_8bit \
    --load_checkpoint $CKPTDIR/$model_name/checkpoint-$CKPTNAME/pytorch_model.bin \
    --per_device_eval_batch_size $bs \
    --output_dir $OUTDIR/fudge \
    --disc_name $disc_name"

elif [[ $exp = 'nado' ]]; then

script="$launcher scripts/run_guided_generation.py \
    $task_options \
    --use_kilt_format False \
    --top_p 0.95 --top_k 50 --temperature 1.0 \
    --model_name google/flan-t5-xl \
    --guidance_method nado \
    --load_8bit True \
    --alpha 0.25 \
    --load_checkpoint $CKPTDIR/$model_name/checkpoint-$CKPTNAME/pytorch_model.bin \
    --per_device_eval_batch_size $bs \
    --output_dir $OUTDIR/nado \
    --disc_name $disc_name"

elif [[ $exp = 'astar' ]]; then
# NOTE: this implmentation is very slow and infeasible.
script="$launcher scripts/run_guided_generation.py \
    $task_options \
    --use_kilt_format False \
    --top_p 0.95 --top_k 50 --temperature 1.0 \
    --model_name google/flan-t5-xl \
    --guidance_method astar \
    --load_8bit \
    --load_checkpoint $CKPTDIR/$model_name/checkpoint-$CKPTNAME/pytorch_model.bin \
    --per_device_eval_batch_size $bs \
    --output_dir $OUTDIR/astar \
    --disc_name $disc_name"

else
    echo not implemented $exp yet.
    exit
fi

if [ $V2 = 1 ]; then
    script="$script --v2"
fi

if [ $human = 1 ]; then
    script="$script --human_indices generations/${task}_human_indices.txt"
fi

if [[ $cont != '' ]]; then
    script="$script --continue_from $cont"
fi

if [ $quick = 1 ]; then
    script="$script --complete_after 10"
fi

eval $script
