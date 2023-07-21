#!/bin/bash

source $HOME/pytorch-venv/bin/activate
export CFDPINN_ROOT="$HOME/pinns"

cfdpinn \
    --debug \
    --no-train \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1/ \
    --load-simulation \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.005 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path $CFDPINN_ROOT/models/tmp.pkl \
    --load-model-path $CFDPINN_ROOT/models/tmp.pt \
    --static-plots