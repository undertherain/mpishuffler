#!/bin/sh
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=shuffle
#SBATCH -t 0-0:3 # time (D-HH:MM) 

#  -output-proctable \

mpirun \
  -np 180 \
  -npernode 10 \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
    python3 ./core.py

