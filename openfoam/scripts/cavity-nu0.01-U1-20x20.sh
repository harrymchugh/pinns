#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc
cd /mnt/openfoam/cases/cavity_nu0.01_U1

for i in 1 2 3 4 5; do
rm -r 0.* [1-9]*
time blockMesh
time icoFoam
done