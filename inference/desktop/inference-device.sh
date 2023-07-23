#!/bin/bash

source $HOME/pytorch-venv/bin/activate
export CFDPINN_ROOT="$HOME/pinns/"
cd $CFDPINN_ROOT

cfdpinn \
    --debug \
    --device cuda \
    --inference-timing \
    --no-train \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1/ \
    --load-simulation \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.001 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path $CFDPINN_ROOT/models/scaler_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pkl \
    --load-model-path $CFDPINN_ROOT/models/model_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pt \
    --static-plots |& tee $CFDPINN_ROOT/inference/desktop/logs/cuda_inf_20x20_withmodel_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.log

cfdpinn \
    --debug \
    --device cpu \
    --no-train \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1/ \
    --load-simulation \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.001 \
    --numx 20 \
    --numy 20 \
    --viscosity 0.01 \
    --initial_u 1 \
    --inference-timing \
    --load-scaler-path $CFDPINN_ROOT/models/scaler_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pkl \
    --load-model-path $CFDPINN_ROOT/models/model_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pt \
    --static-plots |& tee $CFDPINN_ROOT/inference/desktop/logs/cpu_inf_20x20_withmodel_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.log

cfdpinn \
    --debug \
    --device cuda \
    --inference-timing \
    --no-train \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1_100x100/ \
    --load-simulation \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.001 \
    --numx 100 \
    --numy 100 \
    --viscosity 0.01 \
    --initial_u 1 \
    --load-scaler-path $CFDPINN_ROOT/models/scaler_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pkl \
    --load-model-path $CFDPINN_ROOT/models/model_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pt \
    --static-plots |& tee $CFDPINN_ROOT/inference/desktop/logs/cuda_inf_100x100_withmodel_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.log

cfdpinn \
    --debug \
    --device cpu \
    --no-train \
    --case-type cavity \
    --case-dir $CFDPINN_ROOT/openfoam/cases/cavity_nu0.01_U1_100x100/ \
    --load-simulation \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.001 \
    --numx 100 \
    --numy 100 \
    --viscosity 0.01 \
    --initial_u 1 \
    --inference-timing \
    --load-scaler-path $CFDPINN_ROOT/models/scaler_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pkl \
    --load-model-path $CFDPINN_ROOT/models/model_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.pt \
    --static-plots |& tee $CFDPINN_ROOT/inference/desktop/logs/cpu_inf_100x100_withmodel_cavity_nu0.01_U1_20x20_epochs1000_lrannealing_testpcnt0.8.log