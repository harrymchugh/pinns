#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc
mkdir /mnt/cases/cavity_nu0.01_U1_100x100_36ranks
cd /mnt/cases/cavity_nu0.01_U1_100x100_36ranks
rm -r 0.* [1-9]*

time decomposePar
time blockMesh
time mpirun --allow-run-as-root -np 36 icoFoam -parallel
time reconstructPar
