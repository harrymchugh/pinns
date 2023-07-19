#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc
mkdir /mnt/cases/cavity_nu0.01_U1_100x100_2ranks
cd /mnt/cases/cavity_nu0.01_U1_100x100_2ranks
rm -rf processor*
rm -rf 0.* [1-9]*
time blockMesh
time decomposePar
time mpirun --allow-run-as-root -np 2 icoFoam -parallel
time echo
