#!/bin/bash

set -x

#Source openfoam environment
#Not necessary if using container as it is set
#automatically

#Change to openfoam directory
cd ../cases/cavity

#Parallel run or not
export PARALLEL="False"

#Timing reporting
#for each component of cases
if [ $PARALLEL == "True" ];
then
    echo "decomposePar time:"
    time decomposePar
    echo
    echo "blockMesh time:"
    time blockMesh
    echo
    echo "icoFoam time:"
    time icoFoam
    echo
    echo "reconstructPar time:"
    time reconstructPar
    echo
else
    echo "blockMesh time:"
    time blockMesh
    echo
    echo "icoFoam time:"
    time icoFoam
    echo
fi

#Clean up outputs
./Allclean
