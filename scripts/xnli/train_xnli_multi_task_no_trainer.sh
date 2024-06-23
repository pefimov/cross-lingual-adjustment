#!/bin/bash -e

model_path=$1
[ "$model_path" != "" ] || { echo "Specify the model directory!" ; exit 1 ; }
tokenizer_path=$2
[ "$tokenizer_path" != "" ] || { echo "Specify the tokenizer directory!" ; exit 1 ; }
seed=$3
[ "$seed" != "" ] || { echo "Specify the seed!" ; exit 1 ; }
save_path=$4
[ "$save_path" != "" ] || { echo "Specify the data root directory!" ; exit 1 ; }

align_mode=$5
[ "$align_mode" != "" ] || { echo "Specify the align mode!" ; exit 1 ; }
lang=$6
[ "$lang" != "" ] || { echo "Specify the language!" ; exit 1 ; }
sent_path=$7
[ "$sent_path" != "" ] || { echo "Specify the path to parallel sentences!" ; exit 1 ; }
align_path=$8
[ "$align_path" != "" ] || { echo "Specify the path to alignments!" ; exit 1 ; }


python xnli/run_xnli_multi_task_no_trainer.py \
  --model_name_or_path $model_path \
  --tokenizer_name $tokenizer_path \
  --train_language en \
  --language en \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir $save_path \
  --save_steps -1 \
  --seed $seed \
  --align_mode  $align_mode \
  --base_diff_wght 1 \
  --align_wght 0.01 \
  --align_batch_size 16 \
  --sent_path $sent_path \
  --align_path $align_path \
  --include_clssep \
  --align_language $lang \
  --max_sent 30000
