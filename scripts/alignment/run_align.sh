#!/bin/bash -e

model_root_dir=$1
[ "$model_root_dir" != "" ] || { echo "Specify the model root directory!" ; exit 1 ; }
seed=$2
[ "$seed" != "" ] || { echo "Specify the seed!" ; exit 1 ; }
lang=$3
[ "$lang" != "" ] || { echo "Specify the language!" ; exit 1 ; }
data_root_dir=$4
[ "$data_root_dir" != "" ] || { echo "Specify the data root directory!" ; exit 1 ; }
max_sent=$5
[ "$max_sent" != "" ] || { echo "Specify the max_sent!" ; exit 1 ; }
lr=$6
[ "$lr" != "" ] || { echo "Specify the learning rate!" ; exit 1 ; }

echo "lang: $lang max sentence: $max_sent learning rate: $lr"

python alignment/bert_finetuning_alignment.py \
    --model_path $model_root_dir/mbert \
    --tokenizer_path $model_root_dir/mbert \
    --max_sent $max_sent \
    --learning_rate $lr \
    --sent_path $data_root_dir/$lang/wiki_matrix.moses.$lang-en.250k.token \
    --align_path $data_root_dir/$lang/wiki_matrix.moses.$lang-en.250k.intersect \
    --save_path $model_root_dir/aligned_mbert/$lang/$max_sent/$lr/$seed/ \
    --language $lang \
    --include_clssep \
    --align_mode avg \
    --seed $seed


