#!/bin/sh
#$ -cwd
#$ -l f_node=12
#$ -l h_rt=00:05:00
#$ -N shuffle
#$ -m abe
#$ -M alex@smg.is.titech.ac.jp

#  -output-proctable \
source /home/1/drozd-a-aa/apps.sh

mpirun \
  -np 120 \
  -npernode 10 \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
    python3 ./core.py

