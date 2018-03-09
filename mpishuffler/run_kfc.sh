#!/bin/sh
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=shuffle
#SBATCH -t 0-0:4 # time (D-HH:MM) 

#  -output-proctable \

mpirun \
  -np 150 \
  -npernode 10 \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
    python3 ./core2.py

