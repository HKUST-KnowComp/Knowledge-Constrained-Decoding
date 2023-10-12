ids=$1

export CUDA_VISIBLE_DEVICES=$ids

############################## Summarization ##################################
cnn_model_name=(
                baseline/cnn_dailymail-bigscience-T0pp
                baseline/cnn_dailymail-google-flan-t5-xl
                baseline/cnn_dailymail-google-flan-t5-xxl
                baseline/cnn_dailymail-openai_gpt-3.5-turbo
                baseline/cnn_dailymail-openai_text-davinci-003
                fudge/cnn_dailymail-google-flan-t5-xl-fudge-DecoderDisc-cnn_dailymail-RAND
                nado/cnn_dailymail-google-flan-t5-xl-nado-DecoderDisc-cnn_dailymail-ALL-v2-alpha0.25
                ppl_mcts/cnn_dailymail-google-flan-t5-xl-DecoderDisc-cnn_dailymail-RAND
                fudge/cnn_dailymail-google-flan-t5-xl-fudge-DecoderDisc-cnn_dailymail-RIPA
                ppl_mcts/cnn_dailymail-google-flan-t5-xl-DecoderDisc-cnn_dailymail-RIPA
                )

for name in "${cnn_model_name[@]}"; do
    save_name="${name/"/"/-}"

    script="python UniEval/run.py \
            --task summarization \
            --generations_path generations/${name}.jsonl \
            --dataset_path cnn_dailymail \
            --save_name $save_name"
    if [[ $name == *"alpaca"* ]]; then
        script="$script --causal_lm_generations"
    fi

    eval $script

    # MFMA score
    mfma_script="python scripts/evaluate_summary_mfma.py \
        --dataset cnn_dailymail \
        --test_data_path cnn_dailymail \
        --batch_size 8
        --generations_path generations/${name}.jsonl"
    if [[ $name == *"alpaca"* ]]; then
        script="$script --causal_lm_generations"
    fi

    eval $mfma_script

done

############################### WoW Dialogue ###################################
wow_model_name=(
                baseline/wow-openai_gpt-3.5-turbo
                baseline/wow-openai_text-davinci-003
                baseline/wow-bigscience-T0pp
                baseline/wow-google-flan-t5-xl
                baseline/wow-google-flan-t5-xxl
                baseline/wow-google-flan-t5-xl-sft
                fudge/wow-google-flan-t5-xl-fudge-DecoderDisc-wow-RAND
                nado/wow-google-flan-t5-xl-nado-DecoderDisc-wow-ALL-v2-alpha0.25
                ppl_mcts/wow-google-flan-t5-xl-DecoderDisc-wow-RAND
                ppl_mcts/wow-google-flan-t5-xl-DecoderDisc-wow-RIPA
                fudge/wow-google-flan-t5-xl-fudge-DecoderDisc-wow-RIPA
                )

function run_dialog_eval (){
    ids=$1
    name=$2
    skip=$3
    human=$4

    save_name="${name/"/"/-}"

    script="CUDA_VISIBLE_DEVICES=$ids python UniEval/run.py \
            --task dialogue \
            --generations_path generations/${name}.jsonl \
            --dataset_path data/cached/wow/test_unseen.jsonl \
            --save_name $save_name"
    if [[ $name == *"alpaca"* ]]; then
        script="$script --causal_lm_generations"
    fi

    if [[ $skip = 1 ]]; then
        script="$script --skip_no_knowledge"
    fi

    if [[ $human = 1 ]]; then
        script="$script --human_indices generations/wow_human_indices.txt"
    fi

    eval $script
}

for model in "${wow_model_name[@]}"; do
    run_dialog_eval $ids $model 1 0
done
