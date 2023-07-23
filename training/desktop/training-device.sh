#!/bin/bash

source ${HOME}/pytorch-venv/bin/activate
export CFDPINN_ROOT="${HOME}/pinns"

for epochs in 1000; do
for testpct in 0.2; do
for adaption in lrannealing; do
for device in cuda; do

export NUMX="20"
export NUMY="20"
export CAVITY_ROOT="${CFDPINN_ROOT}/openfoam/cases/cavity_nu0.01_U1/"
export CAVITY_NAME_TMP=$(basename $CAVITY_ROOT)
export CAVITY_NAME=${CAVITY_NAME_TMP}_${NUMX}x${NUMY}

export TEST_PERCENT="$testpct"
export EPOCHS="$epochs"
export ADAPTION="$adaption"

cd ${CFDPINN_ROOT}

cfdpinn \
    --device ${device} \
    --debug \
    --tensorboard \
    --tensorboard-path ${CFDPINN_ROOT}/tboard/${device}_${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}/ \
    --case-type cavity \
    --case-dir ${CAVITY_ROOT} \
    --start-time 0 \
    --end-time 5 \
    --sim-dt 0.001 \
    --load-dt 0.005 \
    --numx ${NUMX} \
    --numy ${NUMY} \
    --viscosity 0.01 \
    --initial_u 1 \
    --test-percent ${TEST_PERCENT} \
    --lr 0.001 \
    --epochs ${EPOCHS} \
    --save-scaler-path ${CFDPINN_ROOT}/models/${device}_scaler_${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.pkl \
    --save-model-path ${CFDPINN_ROOT}/models/${device}_model_${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.pt \
    --adaption ${ADAPTION} \
    --prediction-animation \
    --static-plots \
    --residual-animation |& tee training/desktop/logs/${device}_${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.log

done
done
done
done