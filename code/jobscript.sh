#!/bin/sh
#BSUB -q gpua100
#BSUB -J rmlsJob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:00
#BSUB -R "rusage[mem=25GB]"
python3 code/train.py --config="configs/vit.yml"
