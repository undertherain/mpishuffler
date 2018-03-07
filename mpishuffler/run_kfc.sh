#!/bin/sh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=shuffle
#SBATCH -t 0-0:3 # time (D-HH:MM) 

#  -output-proctable \

mpirun \
  -np 16 \
  -npernode 8 \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
    python3 ./core.py

