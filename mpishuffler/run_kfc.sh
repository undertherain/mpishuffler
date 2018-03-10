#!/bin/sh
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=shuffle
#SBATCH -t 0-0:4 # time (D-HH:MM) 

#  -output-proctable \

mpirun \
  -np 200 \
  -npernode 10 \
  -mca btl_openib_ib_timeout 30 \
  -mca btl tcp,sm,self \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
    python3 ./core.py

