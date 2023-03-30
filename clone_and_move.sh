#!/bin/bash

# isntall git lfs
git lfs install

# clone the ckpt
git clone https://huggingface.co/SunnyLee/LongT5_medsum

# Create target file (if not exits)
mkdir -p output/lightning_logs

# move all the files in  LongT5_medsum to output/lightning_logs
mv LongT5_medsum/* output/lightning_logs/

# delete LongT5_medsum 空文件夹
rmdir LongT5_medsum
