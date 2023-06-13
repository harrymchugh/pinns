#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --account=mdisspt-z2137380
#SBATCH --job-name=training-1epoch-20x20
#SBATCH --output=%x.%j.out

module load python/3.10.8-gpu

source /mnt/lustre/indy2lfs/work/mdisspt/mdisspt/z2137380/venvs/cfdpinn-venv/bin/activate

export CFDPINN_ROOT="/work/mdisspt/mdisspt/z2137380/pinns"
cd $CFDPINN_ROOT

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=True

cfdpinn \
    --debug \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1/ \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.005 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --initial_u 1 \
    --test-percent 0.7 \
    --lr 0.001 \
    --epochs 1 \
    --save-scaler-path $CFDPINN_ROOT/models/scaler.pkl \
    --save-model-path $CFDPINN_ROOT/models/2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt
    
