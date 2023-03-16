# MEDIQA-Chat-2023-SZU_Clinical
a private GitHub repo with the team’s code for MEDIQA-Chat-2023-SZU_Clinical
Our code can reproduce the total procedure from trairning to testing. The structure mainly base on pytorch_lightning and our model is LongT5-base. We want to experiment more model like LongT5-large and so on. However, the limitation of memory prevents us from that. I have locally run the code on 4 2080-ti GPU using deepspeed_stage_2.
After traning The pl moudel will produce checkpoin file in folder: output/lightning_log. They have unique version like version_0 and so on. Our code will generate four versions corresponding to four sections


