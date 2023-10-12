ids=$1
type=$2  # [EOS, RIPA, ALL, RAND]
bs=$3
rootdir=$4
CONT=$5
human=$6
guidance=$7
use_t5=$8
chatgpt=$9
debug=${10}

export CUDA_VISIBLE_DEVICES=$ids

if [ $use_t5 = 1 ]; then
    model_name=google/flan-t5-xl
    load_name=flan-t5-xl
else
    model_name=gpt2-xl
    load_name=gpt2-xl
fi

if [ $chatgpt = 1 ]; then
    openai_model_name=gpt-3.5-turbo
else
    openai_model_name=text-davinci-003
fi

script="python scripts/run_openai_guided_generation.py \
    --model_name $model_name \
    --openai_model_name $openai_model_name \
    --num_labels 2 \
    --dataset wow \
    --test_data_path $rootdir/data/cached/wow/test_unseen.jsonl \
    --output_path $rootdir/generations/fudge \
    --guidance_method openai_fudge \
    --instruction_model basic \
    --load_checkpoint $rootdir/saved_models/$load_name-DecoderDisc-wow-$type/checkpoint-best/pytorch_model.bin \
    --batch_size $bs"

disc_name=$openai_model_name-fudge-$type

if [ $debug = 1 ]; then
    script="$script --mock_debug"
fi

if [ $human = 1 ]; then
    script="$script --human_indices $rootdir/generations/wow_human_indices.txt"
fi

if [[ $CONT != 0 ]]; then
    script="$script --continue_from $CONT"
fi

if [ $guidance = 1 ]; then
    script="$script --use_logit_bias True --propose_topk 50"
    disc_name=$disc_name-logit_bias
elif [ $guidance = 2 ]; then
    script="$script --use_logit_bias False"
    disc_name=$disc_name-post_guidance
elif [ $guidance = 3 ]; then
    script="$script --use_logit_bias True --propose_topk 50 --pre_post_guidance"
    disc_name=$disc_name-pre_post_guidance
fi

script="$script --disc_name $disc_name"

eval $script
