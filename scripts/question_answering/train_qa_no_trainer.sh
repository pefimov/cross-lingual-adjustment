#!/bin/bash -e

model_path=$1
[ "$model_path" != "" ] || { echo "Specify the model directory!" ; exit 1 ; }
tokenizer_path=$2
[ "$tokenizer_path" != "" ] || { echo "Specify the tokenizer directory!" ; exit 1 ; }
seed=$3
[ "$seed" != "" ] || { echo "Specify the seed!" ; exit 1 ; }
save_path=$4
[ "$save_path" != "" ] || { echo "Specify the data root directory!" ; exit 1 ; }



python3 question_answering/run_qa_no_trainer.py \
  --model_name_or_path $model_path \
  --tokenizer_name $tokenizer_path \
  --dataset_name squad \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed $seed \
  --output_dir $save_path