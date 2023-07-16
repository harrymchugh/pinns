#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --account=mdisspt-z2137380
#SBATCH --time=00:20:00
#SBATCH --job-name=sing-train-v100-500epoch-20x20_0_5_0.005_lr0.001_test0.7_adam
#SBATCH --output=%x.%j.out

module load nvidia/nvhpc/22.11
module load singularity

srun singularity run --nv -B /work/mdisspt/mdisspt/z2137380/pinns:/mnt \
/work/mdisspt/mdisspt/z2137380/cfdpinn_latest.sif \
cfdpinn \
    --debug \
    --profile \
    --profile-path /mnt/profiles/tboard/ \
    --tensorboard \
    --tensorboard-path /mnt/tboard/ \
    --case-type cavity \
    --case-dir /mnt/openfoam/cases/cavity_nu0.01_U1/ \
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
    --epochs 500 \
    --save-scaler-path /mnt/models/tmp-gpu.pkl \
    --save-model-path /mnt/models/tmp-gpu.pt
    
