ids=$1
typ=$2
metric=$3
task=$4
bs=$5
CLASS_ONLY=$6
V2=$7
CONT=$8
human=$9
quick_option=${10}

if [ $bs = '' ]; then
    bs=16
fi

if [ $CONT = '' ]; then
    CONT=0
fi

DATADIR=data/cached
OUTDIR=generations
CKPTDIR=saved_models
CKPTNAME=best

if [ $task = 'wow' ]; then
    task_options="--test_data_path $DATADIR/wow/test_unseen.jsonl --dataset wow --max_new_tokens 32"
elif [ $task = 'cnn_dailymail' ]; then
    task_options="--test_data_path cnn_dailymail --dataset cnn_dailymail --max_new_tokens 64"
else
    echo $task not defined.
    exit
fi

model_name=flan-t5-xl-DecoderDisc-$task-$typ
disc_name=DecoderDisc-$task-$typ

if [ $V2 = 1 ]; then
    model_name=$model_name-v2
    disc_name=$disc_name-v2
fi


if [ $CLASS_ONLY = 1 ]; then
    ckpt_options="--use_mlp_classifier --load_classifier $CKPTDIR/$model_name-only_classifier/checkpoint-$CKPTNAME/pytorch_model.bin"
else
    ckpt_options="--load_checkpoint $CKPTDIR/$model_name/checkpoint-$CKPTNAME/pytorch_model.bin"
fi

copy_penalty=1.0
script="CUDA_VISIBLE_DEVICES=$ids python scripts/run_ppl_mcts.py \
    $task_options \
    --use_kilt_format False \
    --lm_name google/flan-t5-xl \
    --num_labels 2 --attr_idx 1 \
    --load_8bit \
    $ckpt_options \
    --output_path $OUTDIR/ppl_mcts \
    --batch_size $bs \
    --num_simulations 50 \
    --knowledge_copy_penalty $copy_penalty \
    --top_k 50 \
    --temperature 1.0 \
    --continue_from $CONT"


if [[ $metric != '' ]]; then
    script="$script --guide_using_metric True --metric_name $metric --disc_name $metric"
else
    script="$script --disc_name $disc_name"
fi

if [ $V2 = 1 ]; then
    script="$script --v2"
fi

if [ $human = 1 ]; then
    script="$script --human_indices generations/${task}_human_indices.txt"
fi

if [[ $quick_option != 0 ]]; then
    # this will change everytime
    script="$script --complete_after $quick_option"
fi

eval $script
