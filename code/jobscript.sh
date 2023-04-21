#!/bin/sh
#BSUB -q gpua100
#BSUB -J rmlsJob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 18:00
#BSUB -R "rusage[mem=20GB]"
python3 code/train_transformer.py --model="chex2"
