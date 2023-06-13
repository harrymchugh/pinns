#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --account=mdisspt-z2137380
#SBATCH --time=00:20:00
#SBATCH --job-name=train-gpu-1000epoch-20x20_0_5_0.005
#SBATCH --output=%x.%j.out

module load python/3.10.8-gpu
source /mnt/lustre/indy2lfs/work/mdisspt/mdisspt/z2137380/venvs/cfdpinn-venv/bin/activate
export CFDPINN_ROOT="/work/mdisspt/mdisspt/z2137380/pinns"
cd $CFDPINN_ROOT

mkdir -p $CFDPINN_ROOT/tboard

cfdpinn \
    --debug \
    --tensorboard \
    --tensorboard-path $CFDPINN_ROOT/tboard/ \
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
    --epochs 1000 \
    --save-scaler-path $CFDPINN_ROOT/models/cavity_nu0.01_U1_20x20_0_5_0.005_scaler.pkl \
    --save-model-path $CFDPINN_ROOT/models/2d-ns-pinn-1000epochs-lr0.001-adam-cavity-nu0.01-u1-20x20_0_5_0.005.pt
    
