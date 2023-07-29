#!/bin/bash

cfdpinn \
    --debug \
    --no-train \
    --case-type cavity \
    --case-dir /mnt/openfoam/cases/cavity_nu0.01_U1/ \
    --load-simulation \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.005 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path /mnt/models/scaler_001.pkl \
    --load-model-path /mnt/models/model_001.pt \
    --static-plots \
    --static-plots-path /mnt/plots/
