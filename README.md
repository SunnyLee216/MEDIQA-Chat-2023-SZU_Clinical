# MEDIQA-Chat-2023-SZU_Clinical
a private GitHub repo with the team’s code for MEDIQA-Chat-2023-SZU_Clinical Our code can reproduce the total procedure from trairning to testing. The structure mainly base on pytorch_lightning and our model is LongT5-base. We want to experiment more model like LongT5-large and so on. However, the limitation of memory prevents us from that. I have locally run the code on 4 2080-ti GPU using deepspeed_stage_2. After traning The pl moudel will produce checkpoin file in folder: output/lightning_log. They have unique version like version_0 and so on. Our code will generate four versions corresponding to four sections

However, due to my negligence, I had only conducted experiments in a conda environment before. My experiments mainly used deepspeed, but I found that it could not be installed successfully via a Python virtual environment, so install.sh failed to install deepspeed, though I have completed the experiments on my local conda environment. Our code is simply fine-tuned and the results are reliable.It's easy to reproduce the outcome on a conda environment. Unfortunately, due to time constraints, I am unable to modify the code. I am very sorry.

error raise, when pip install deepspeed:

error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
 We only complete the taskB, the procedure shout be:

```
bash ./install.sh
source ./activate.sh
bash decode_taskB_run1.sh ./MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset.csv
bash decode_taskB_run2.sh ./MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset.csv
bash decode_taskB_run3.sh ./MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset.csv

```

the [./MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset.csv] is where I put the Testset



