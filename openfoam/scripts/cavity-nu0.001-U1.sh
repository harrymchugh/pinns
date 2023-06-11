#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc
cd /mnt/cases/cavity_nu0.001_U1
rm -r 0.* [1-9]*
time blockMesh
time icoFoam

#Get number of CPUs available
#lscpu | grep "^CPU(s):" | sed -e 's/\s\+/,/g' | cut -f 2 -d ","
#
#
