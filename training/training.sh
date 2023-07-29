#!/bin/bash

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
    --epochs 5 \
    --save-scaler-path /mnt/models/tmp.pkl \
    --save-model-path /mnt/models/tmp.pt

