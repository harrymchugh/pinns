#!/bin/bash

set -x

export cavity_base="$HOME/pinns/openfoam/cases/cavity"

if [[ "$OSTYPE" == "darwin"* ]]; then
    export SED_ARGS="gsed -i"
else
    export SED_ARGS="sed -i"
fi

echo $SED_ARGS

for nu in 0.01 0.001 0.0001; do
    for velocity in 1 4 8; do
        #Copy base case
        cp -r "$cavity_base" "$cavity_base"_nu"$nu"_U"$velocity"

        #Update viscosity
        $SED_ARGS "s/0.01/$nu/g" "$cavity_base"_nu"$nu"_U"$velocity"/constant/transportProperties

        #Update velocity
        $SED_ARGS "s/(1 0 0)/($velocity 0 0)/g" "$cavity_base"_nu"$nu"_U"$velocity"/0/U
    done
done

