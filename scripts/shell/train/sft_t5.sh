##### stdin ####
ids=$1
################
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

dataset=wow
use_kilt_format=False
size=xl

if [[ $use_kilt_format = True ]]; then
    model_name=flan-t5-$size-sft-$dataset-kilt
else
    model_name=flan-t5-$size-sft-$dataset
fi
lr=1e-5
bs=16
grad_accum=2


script="$launcher kcd/token_classifier/train.py \
    --sft \
    --model_name google/flan-t5-$size \
    --is_decoder \
    --wandb_project_name knowledge-sft \
    --wandb_run_name $model_name \
    --dataset $dataset \
    --use_kilt_format $use_kilt_format \
    --train_data_path data/cached/wow/train.jsonl \
    --validation_data_path data/cached/wow/dev_unseen.jsonl \
    --output_dir saved_models/$model_name \
    --use_lora --bf16 --train_8bit \
    --learning_rate $lr \
    --warmup_steps 0 \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --max_steps 2000 \
    --logging_steps 10 \
    --eval_accumulation_steps 100 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --load_best_model_at_end False \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $grad_accum"

eval $script