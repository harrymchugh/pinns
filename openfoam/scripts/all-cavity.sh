#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc

cd /mnt/cases

for dir in cavity_nu*; do
    cd $dir
    rm -r 0.* [1-9]*
    time blockMesh
    time icoFoam
    cd ../
done

#Get number of CPUs available
#lscpu | grep "^CPU(s):" | sed -e 's/\s\+/,/g' | cut -f 2 -d ","
#
#
