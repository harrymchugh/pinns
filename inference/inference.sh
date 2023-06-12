#!/bin/bash

#Macbook pro
export VENV="$HOME/cfdpinn-venv/bin/activate" 
#Work laptop
#export VENV="$HOME/pinns-venv/bin/activate"
#Cirrus
#export VENV="$HOME/pinns-venv/bin/activate"

export OMP_NUM_THREADS=$1
export OMP_PROC_BIND=True

cfdpinn \
    --debug \
    --profile \
    --trace-path $HOME/pinns/profiles/100x100_0_5_0.001_trace.json \
    --stack-path $HOME/pinns/profiles/100x100_0_5_0.001_stk.txt \
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
    --load-scaler-path $HOME/pinns/models/tmp-scaler.pkl \
    --load-model-path $HOME/pinns/models/tmp-2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt
    
