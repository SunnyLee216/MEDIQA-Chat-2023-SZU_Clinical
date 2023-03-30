#!/bin/bash

# isntall git lfs
git lfs install

# clone the ckpt
git clone https://huggingface.co/SunnyLee/LongT5_medsum

# 创建目标目录（如果不存在）
mkdir -p output/lightning_logs

# 将 LongT5_medsum 文件夹中的所有文件移动到 output/lightning_logs
mv LongT5_medsum/* output/lightning_logs/

# 删除 LongT5_medsum 空文件夹
rmdir LongT5_medsum