#!/bin/bash

#Macbook pro
#export VENV="$HOME/cfdpinn-venv/bin/activate" 
#Work laptop
#export VENV="$HOME/pinns-venv/bin/activate"
#Cirrus
#export VENV="$HOME/pinns-venv/bin/activate"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=True

cfdpinn \
    --debug \
    --no-train \
    --case-type cavity \
    --case-dir $HOME/pinns/openfoam/cases/cavity_nu0.01_U1_100x100/ \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.001 \
    --numx 100 \
    --numy 100 \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path $WORK/pinns/models/scaler.pkl \
    --load-model-path $WORK/pinns/models/2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt
    
