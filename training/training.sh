#!/bin/bash

cfdpinn \
    --debug \
    --profile \
    --profile-path /mnt/profiles/tboard/model_001/ \
    --tensorboard \
    --tensorboard-path /mnt/tboard/model_001/ \
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
    --epochs 1 \
    --save-scaler-path /mnt/models/scaler_001.pkl \
    --save-model-path /mnt/models/model_001.pt \
    --training-animation \
    --prediction-animation \
    --residual-animation \
    --animations-path /mnt/plots/ \
    --static-plots \
    --static-plots-path /mnt/plots/

