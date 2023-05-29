#!/bin/bash

set -x

#Source openfoam environment
#Not necessary if using container as it is set
#automatically
APPTAINER_SIF="../../containers/apptainer/cfdpinn.sif"

#Change to openfoam directory
cd ../cases/cavity

#Parallel run or not
export PARALLEL="False"

#Timing reporting
#for each component of cases
if [ $PARALLEL == "True" ];
then
    echo "decomposePar time:"
    time apptainer run --no-home -B $PWD:/cavity -W /cavity $APPTAINER_SIF decomposePar
    echo
    echo "blockMesh time:"
    time apptainer run --no-home -B $PWD:/cavity -W /cavity $APPTAINER_SIF blockMesh
    echo
    echo "icoFoam time:"
    time apptainer run --no-home -B $PWD:/cavity -W /cavity $APPTAINER_SIF icoFoam
    echo
    echo "reconstructPar time:"
    time apptainer run --no-home -B $PWD:/cavity -W /cavity $APPTAINER_SIF reconstructPar
    echo
else
    echo "blockMesh time:"
    time apptainer run --no-home -B $PWD:/cavity -W /cavity $APPTAINER_SIF blockMesh
    echo
    echo "icoFoam time:"
    time apptainer run --no-home -B $PWD:/cavity -W /cavity $APPTAINER_SIF icoFoam
    echo
fi
