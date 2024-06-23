#!/bin/bash -e

model_path=$1
[ "$model_path" != "" ] || { echo "Specify the model directory!" ; exit 1 ; }
tokenizer_path=$2
[ "$tokenizer_path" != "" ] || { echo "Specify the tokenizer directory!" ; exit 1 ; }
lang=$3
[ "$lang" != "" ] || { echo "Specify the language!" ; exit 1 ; }
save_path=$4
[ "$save_path" != "" ] || { echo "Specify the data root directory!" ; exit 1 ; }


python3 xnli/run_xnli.py \
  --model_name_or_path $model_path \
  --tokenizer_name $tokenizer_path \
  --language $lang \
  --do_predict \
  --max_seq_length 128 \
  --output_dir $save_path \