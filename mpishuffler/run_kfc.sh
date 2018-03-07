#!/bin/sh
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=shuffle
#SBATCH -t 0-0:20 # time (D-HH:MM) 

#  -output-proctable \

mpirun \
  -np 200 \
  -npernode 8 \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
    python3 ./core.py

