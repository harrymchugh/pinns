#!/bin/bash

source /home/hamchugh/pinns-venv/bin/activate

cfdpinn \
    --case-type cavity \
    --case-dir $HOME/pinns/openfoam/cases/cavity/ \
    --start-time 0 \
    --end-time 5 \
    --dt 0.005 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --initial_u 1 \
    --model-path $HOME/pinns/models/tmp-2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt\
    --epochs 1 \
    --training-animation \
    --prediction-animation \
    --residual-animation \
    --num-frames 5 \
    --inference-timing \
    --output-raw-data \
    --output-train-data \
    --output-data-path $HOME/pinns/data/tmp-outputs
    
