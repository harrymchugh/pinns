#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc
cd /mnt/cases/cavity
rm -r 0.* [1-9]*
time blockMesh
time icoFoam
