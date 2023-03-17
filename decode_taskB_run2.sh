#!/bin/bash

# 检查输入参数是否为空
if [ -z "$1" ]
  then
    echo "Please provide the input-csv-file as an argument."
    exit 1
fi

# 设置变量
input_csv=$1
output_csv="./outputs_run/taskB_SZU_Clinical_run2.csv"
# HPI
python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 2700 --is_split "HISTORY OF PRESENT ILLNESS" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/vesion_0/checkpoints/best.ckpt --test_file $input_csv  

# # # PHYSICAL 

python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 1100 --is_split "PHYSICAL EXAM" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/vesion_1/checkpoints/best.ckpt --test_file $input_csv

# # # RESULTS 
python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 900 --is_split "RESULTS" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/vesion_2/checkpoints/best.ckpt --test_file $input_csv

# # # ASSESSMENT

python long_training.py --gpus 1 --strategy deepspeed_stage_2 --precision bf16 --max_input_length 10000 --max_output_length 2700 --is_split "ASSESSMENT AND PLAN" --pretrained_model google/long-t5-tglobal-base --is_test True --checkpoint ./output/lightning_logs/vesion_3/checkpoints/best.ckpt --test_file $input_csv

python merge.py $input_csv $output_csv