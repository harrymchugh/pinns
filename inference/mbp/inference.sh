#!/bin/bash

source $HOME/cfdpinn-venv/bin/activate
export CFDPINN_ROOT="$HOME/pinns"
cd $CFDPINN_ROOT

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=True

cfdpinn \
    --debug \
    --profile \
    --trace-path $CFDPINN_ROOT/profiles/20x20_0_5_0.005_trace.json \
    --stack-path $CFDPINN_ROOT/profiles/20x20_0_5_0.005_stk.txt \
    --no-train \
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
    --load-scaler-path $CFDPINN_ROOT/models/scaler.pkl \
    --load-model-path $CFDPINN_ROOT/models/tmp-2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt
    
