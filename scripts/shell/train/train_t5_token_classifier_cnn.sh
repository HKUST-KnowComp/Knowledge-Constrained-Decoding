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

launcher="CUDA_VISIBLE_DEVICES=$ids python"

dataset=cnn_dailymail
size=xl
if [[ $type = '' ]]; then
    echo type was not provided. Defaults to ALL...
    type=ALL
fi
if [[ $type = EOS ]]; then
    pool='last'
elif [[ $type = RAND ]]; then
    pool='random'
else
    pool='none'
fi


model_name=flan-t5-$size-DecoderDisc-$dataset-$type
train_data_path=data/cached/cnn_dailymail_train_augmented_neg_google-flan-t5-xl-0.9
validation_data_path=data/cached/cnn_dailymail_test_augmented_neg_google-flan-t5-xl-0.1

if [ $ONLY_CLASS = 1 ]; then
    model_name=$model_name-only_classifier
fi

if [ $V2 = 1 ]; then
    model_name=$model_name-v2
    if [[ $V2_REG != 0 ]]; then
        model_name=$model_name-v2reg$V2_REG
    fi
fi

lr=5e-6
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
    --use_kilt_format False \
    --train_data_path $train_data_path \
    --validation_data_path $validation_data_path \
    --output_dir saved_models/$model_name \
    --bf16 --train_8bit \
    --learning_rate $lr \
    --warmup_steps 0 \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --max_steps 2000 \
    --logging_steps 10 \
    --eval_accumulation_steps 50 \
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
    script="$script --load_checkpoint saved_models/flan-t5-$size-DecoderDisc-$dataset-EOS/checkpoint-best/pytorch_model.bin"
fi

eval $script
