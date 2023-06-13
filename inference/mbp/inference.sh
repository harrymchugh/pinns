#!/bin/bash

source $HOME/cfdpinn-venv/bin/activate
export CFDPINN_ROOT="$HOME/pinns"
cd $CFDPINN_ROOT/inference/mbp

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=True

for numx in 20 40 80; do
    for dt in 0.001 0.002; do

cfdpinn \
    --debug \
    --profile \
    --trace-path $CFDPINN_ROOT/profiles/"$numx"x"$numx"_0_5_"$dt"_trace.json \
    --stack-path $CFDPINN_ROOT/profiles/"$numx"x"$numx"_0_5_"$dt"_stk.txt \
    --no-train \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt $dt \
    --numx $numx \
    --numy $numx \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path $CFDPINN_ROOT/models/scaler.pkl \
    --load-model-path $CFDPINN_ROOT/models/tmp-2d-ns-pinn-1epochs-lr0.001-adam-cavity-nu0.01-u1-20x20.pt > inf-"$numx"x"$numx"_0_5_"$dt".log
    
    done
done