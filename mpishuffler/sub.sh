if [[ $SGE_CLUSTER_NAME = "t3" ]]; then
    # TSUBAME 3.0
    qsub -g tgc-ebdcrest run_t3.sh
else
    # probably TSUBAME-KFC/DL
    sbatch run_kfc.sh
fi
