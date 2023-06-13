#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --account=mdisspt-z2137380
#SBATCH --job-name=inference-100x100-cpu-jemalloc
#SBATCH --output=%x.%j.out
#SBATCH --exclusive

module load python/3.10.8-gpu

source /mnt/lustre/indy2lfs/work/mdisspt/mdisspt/z2137380/venvs/cfdpinn-venv/bin/activate

export CFDPINN_ROOT="/work/mdisspt/mdisspt/z2137380/pinns"
cd $CFDPINN_ROOT

export LD_PRELOAD="/mnt/lustre/indy2lfs/work/mdisspt/mdisspt/z2137380/libjemalloc.so"

for threads in 1; do

export OMP_NUM_THREADS=$threads
export OMP_PROC_BIND=True

numactl --cpunodebind=0 --membind=0 cfdpinn \
    --debug \
    --no-train \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1_100x100/ \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.001 \
    --numx 100 \
    --numy 100 \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path $CFDPINN_ROOT/models/scaler.pkl \
    --load-model-path $CFDPINN_ROOT/models/tmp-2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt |& tee $CFDPINN_ROOT/inference/cirrus/inf-100x100-0-5-0.001-jemalloc-threads$threads.log

done
    
