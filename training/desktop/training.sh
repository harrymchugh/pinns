#!/bin/bash

source ${HOME}/pytorch-venv/bin/activate
export CFDPINN_ROOT="${HOME}/pinns"

for epochs in 200 1000 5000; do
for testpct in 0.2 0.5 0.8; do
for adaption in softadapt lrannealing noadaption; do

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
    --debug \
    --tensorboard \
    --tensorboard-path ${CFDPINN_ROOT}/tboard/${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}/ \
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
    --save-scaler-path ${CFDPINN_ROOT}/models/scaler_${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.pkl \
    --save-model-path ${CFDPINN_ROOT}/models/model_${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.pt \
    --adaption ${ADAPTION} \
    --prediction-animation \
    --static-plots \
    --residual-animation |& tee training/desktop/logs/${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.log

mv \
    ${CFDPINN_ROOT}/animations/tmp_pred.mp4 \
    ${CFDPINN_ROOT}/animations/${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}_prediction.mp4

mv \
    ${CFDPINN_ROOT}/plots/static.png \
    ${CFDPINN_ROOT}/plots/${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}.png

mv \
    ${CFDPINN_ROOT}/animations/tmp_residual.mp4 \
    ${CFDPINN_ROOT}/animations/${CAVITY_NAME}_epochs${EPOCHS}_${ADAPTION}_testpcnt${TEST_PERCENT}_residual.mp4

done
done
done