##### stdin ####
ids=$1
type=$2   # [EOS, ALL, RAND, RIPA]
ONLY_CLASS=$3
V2=$4
V2_REG=$5
FINETUNE=$6
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

train_data_path=$DATADIR/wow_train_augmented
validation_data_path=$DATADIR/wow_dev_unseen_augmented

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
    model_name=flan-t5-$size-DecoderDisc-$dataset-kilt-$type
else
    model_name=flan-t5-$size-DecoderDisc-$dataset-$type
fi
if [ $ONLY_CLASS = 1 ]; then
    model_name=$model_name-only_classifier
fi
if [ $V2 = 1 ]; then
    model_name=$model_name-v2
    if [[ $V2_REG != 0 ]]; then
        model_name=$model_name-v2reg$V2_REG
    fi
fi

lr=1e-5
bs=8
grad_check=False
grad_accum=4


script="$launcher kcd/token_classifier/train.py \
    --model_name google/flan-t5-$size \
    --is_decoder \
    --num_labels 2 \
    --pool_method $pool \
    --wandb_project_name knowledge-classifier \
    --wandb_run_name $model_name \
    --dataset $dataset \
    --use_kilt_format $use_kilt_format \
    --train_data_path $train_data_path \
    --validation_data_path $validation_data_path \
    --output_dir $CKPTDIR/$model_name \
    --bf16 --train_8bit \
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
    --gradient_checkpointing $grad_check"

if [[ $type = ALL ]]; then
    script="$script --sequence_label"
fi


if [ $ONLY_CLASS = 1 ]; then
    script="$script --only_classifier --use_mlp_classifier"
else
    script="$script --use_lora"
fi

if [ $V2 = 1 ]; then
    # regularization default in NADO = 0.5
    script="$script --v2 --nado_reg $V2_REG"
fi

if [ $FINETUNE = 1 ]; then
    script="$script --load_checkpoint $CKPTDIR/flan-t5-$size-DecoderDisc-$dataset-EOS/checkpoint-$CKPTNAME/pytorch_model.bin"
fi

eval $script
