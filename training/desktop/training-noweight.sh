#!/bin/bash

source $HOME/pytorch-venv/bin/activate
export CFDPINN_ROOT="$HOME/pinns"
cd $CFDPINN_ROOT

cfdpinn \
    --debug \
    --tensorboard \
    --tensorboard-path $CFDPINN_ROOT/tboard/noweight/ \
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
    --epochs 200 \
    --save-scaler-path $CFDPINN_ROOT/models/tmp.pkl \
    --save-model-path $CFDPINN_ROOT/models/tmp.pt \
    --adaption noadaption \
    --prediction-animation

mv $CFDPINN_ROOT/animations/tmp_pred.mp4 $CFDPINN_ROOT/animations/tmp_pred_noweight.mp4