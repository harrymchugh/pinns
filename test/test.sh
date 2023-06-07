#!/bin/bash

source /home/hamchugh/pinns-venv/bin/activate

cfdpinn \
    --case-dir /home/hamchugh/pinns/data/cavity/ \
    --start-time 0 \
    --end-time 5 \
    --dt 0.005 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --model-path /home/hamchugh/pinns/models/tmp-2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt\
    --epochs 10 \
    --training-animation \
    --prediction-animation \
    --residual-animation
    
