data=$1


if [ $data = 'wow' ]; then
    data_options="--dataset_name wow --dataset_path data/cached/wow/train.jsonl \
                  --use_kilt_format False"
elif [ $data = 'cnn_dailymail' ]; then
    data_options="--dataset_path cnn_dailymail --dataset_name cnn_dailymail \
                  --use_kilt_format False"
else
    echo $data not recognized.
    exit
fi

script="python kcd/sample_negative.py $data_options"
eval $script



