#!/bin/bash -e

model_path=$1
[ "$model_path" != "" ] || { echo "Specify the model directory!" ; exit 1 ; }
tokenizer_path=$2
[ "$tokenizer_path" != "" ] || { echo "Specify the tokenizer directory!" ; exit 1 ; }
dataset_path=$3
[ "$dataset_path" != "" ] || { echo "Specify the dataset path!" ; exit 1 ; }
tgt_language=$4
[ "$tgt_language" != "" ] || { echo "Specify the target language!" ; exit 1 ; }
specific_layer=$5
[ "$specific_layer" != "" ] || { echo "Specify the layer for embeddings extraction!" ; exit 1 ; }
distance=$6
[ "$distance" != "" ] || { echo "Specify the distance (cosine or l2)!" ; exit 1 ; }
pool_type=$7
[ "$pool_type" != "" ] || { echo "Specify the pooling type (mean or cls)!" ; exit 1 ; }
save_path=$8
[ "$save_path" != "" ] || { echo "Specify the save directory!" ; exit 1 ; }



python retrieval/evaluate_retrieval.py \
    --embed_size 768 \
    --pool_type $pool_type \
    --model_type "bert" \
    --model_name_or_path $model_path \
    --src_language en \
    --tgt_language $tgt_language \
    --batch_size 100 \
    --dataset_path $dataset_path \
    --num_layers 12 \
    --dist $distance \
    --output_dir $save_path \
    --log_file logs \
    --tokenizer_name $tokenizer_path \
    --max_seq_length 512 \
    --specific_layer $specific_layer \
