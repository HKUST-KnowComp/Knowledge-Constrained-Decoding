##### stdin ####
ids=$1
type=$2   # [EOS, ALL, RAND, RIPA]
ONLY_CLASS=$3
FINETUNE=$4
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

DATADIR=data/cached
CKPTDIR=saved_models
CKPTNAME=best


dataset=wow
use_kilt_format=False
size=xl

train_data_path=$DATADIR/wow_train_augmented_neg_google-flan-t5-xl-0.9+random
validation_data_path=$DATADIR/wow_test_augmented_neg_google-flan-t5-xl-0.1+random

if [[ $type = '' ]]; then
    echo type was not provided. Defaults to ALL...
    type=ALL
fi
if [[ $type = EOS ]]; then
    pool='last'
elif [[ $type = RAND ]]; then
    pool='random'
elif [[ $type = RIPA ]]; then
    pool='none'
else
    # ALL
    pool='none'
fi
if [[ $use_kilt_format = True ]]; then
    model_name=gpt2-$size-DecoderDisc-$dataset-kilt-$type
else
    model_name=gpt2-$size-DecoderDisc-$dataset-$type
fi
if [ $ONLY_CLASS = 1 ]; then
    model_name=$model_name-only_classifier
fi

lr=1e-5
bs=32
grad_check=True
grad_accum=1


script="$launcher kcd/token_classifier/train.py \
    --model_name gpt2-$size \
    --num_labels 2 \
    --is_decoder False \
    --pool_method $pool \
    --wandb_project_name knowledge-classifier \
    --wandb_run_name $model_name \
    --dataset $dataset \
    --use_kilt_format $use_kilt_format \
    --train_data_path $train_data_path \
    --validation_data_path $validation_data_path \
    --output_dir $CKPTDIR/$model_name \
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
    --gradient_accumulation_steps $grad_accum \
    --gradient_checkpointing $grad_check --bf16"

if [[ $type = ALL ]]; then
    script="$script --sequence_label"
fi


if [ $ONLY_CLASS = 1 ]; then
    script="$script --only_classifier --use_mlp_classifier"
else
    script="$script --use_lora"
fi

if [ $FINETUNE = 1 ]; then
    script="$script --load_checkpoint $CKPTDIR/gpt2-$size-DecoderDisc-$dataset-EOS/checkpoint-$CKPTNAME/pytorch_model.bin"
fi

eval $script
