##### stdin ####
ids=$1
type=$2   # [EOS, ALL, RAND, RIPA]
dataset=$3
bs=$4
V2=$5
pool=$6
################
# compute number of gpus
arrIDs=(${ids//,/ })
GPU_PER_NODE="${#arrIDs[@]}"

# decide python launcher
if [ $ids = '' ]; then
    echo "no gpu..."
    launcher="CUDA_VISIBLE_DEVICES= python"
elif [ $GPU_PER_NODE = 1 ]; then
    echo "Using 1 GPU: use simple python launcher..."
    launcher="CUDA_VISIBLE_DEVICES=$ids python"
else
    echo "Using multi-GPU: using torchrun launcher..."
    launcher="CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE"
fi

if [[ $bs = '' ]]; then
    bs=64
fi

use_kilt_format=False
size=xl
if [[ $type = '' ]]; then
    echo type was not provided. Defaults to RIPA...
    type=RIPA
fi
if [[ $use_kilt_format = True ]]; then
    model_name=flan-t5-$size-DecoderDisc-$dataset-kilt-$type
else
    model_name=flan-t5-$size-DecoderDisc-$dataset-$type
fi
if [ $V2 = 1 ]; then
    model_name=$model_name-v2
fi

if [ $dataset = wow ]; then
    validation_data_path=data/cached/wow_test_unseen_augmented_neg_random
    test_data_path=data/cached/wow_test_augmented_neg_google-flan-t5-xl-0.1
elif [ $dataset = cnn_dailymail ]; then
    validation_data_path=data/cached/cnn_dailymail_test_augmented_neg_google-flan-t5-xl-0.1
    test_data_path=data/cached/cnn_dailymail_test_augmented_neg_google-flan-t5-xl-0.1
else
    echo $dataset unknown.
    exit
fi

script="$launcher kcd/token_classifier/train.py \
    --model_name google/flan-t5-$size \
    --is_decoder \
    --num_labels 2 \
    --pool_method $pool \
    --wandb_project_name knowledge-classifier \
    --wandb_run_name eval_$model_name-$pool \
    --dataset $dataset \
    --use_kilt_format $use_kilt_format \
    --train_data_path $validation_data_path \
    --validation_data_path $validation_data_path \
    --test_data_path $test_data_path \
    --test_only \
    --load_checkpoint saved_models/$model_name/checkpoint-best/pytorch_model.bin \
    --output_dir saved_models/$model_name \
    --use_lora --train_8bit \
    --eval_accumulation_steps 100 \
    --per_device_eval_batch_size $bs"

if [ $V2 = 1 ]; then
    script="$script --v2"
fi

eval $script
