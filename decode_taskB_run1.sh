#!/bin/bash

# 检查输入参数是否为空
if [ -z "$1" ]
  then
    echo "Please provide the input-csv-file as an argument."
    exit 1
fi

# 设置变量
input_csv=$1
output_csv="./outputs_run/taskB_SZU_Clinical_run1.csv"
# ## train split_HPI
# python long_training.py --gpus 4 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 2500 --is_split "HISTORY OF PRESENT ILLNESS" --pretrained_model google/long-t5-tglobal-base
# python long_training.py --gpus 4 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 1000 --is_split "PHYSICAL EXAM" --pretrained_model google/long-t5-tglobal-base
# python long_training.py --gpus 4 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 800 --is_split "RESULTS" --pretrained_model google/long-t5-tglobal-base
# python long_training.py --gpus 4 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 2500 --is_split "ASSESSMENT AND PLAN" --pretrained_model google/long-t5-tglobal-base


## train split_PHYSICAL EXAM

# HPI
python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 2200 --is_split "HISTORY OF PRESENT ILLNESS" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/version_0/checkpoints/best.ckpt --test_file $input_csv  

# PHYSICAL 
python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 1000 --is_split "PHYSICAL EXAM" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/version_1/checkpoints/best.ckpt --test_file $input_csv

# RESULTS 
python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 800 --is_split "RESULTS" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/version_2/checkpoints/best.ckpt --test_file $input_csv

# ASSESSMENT
python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 1800 --is_split "ASSESSMENT AND PLAN" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/version_3/checkpoints/best.ckpt --test_file $input_csv

# MERGE 4
python merge.py $input_csv $output_csv