#!/usr/bin/env bash

set -e

mkdir -p build
cd build
git clone --branch=mpi-continue-master --depth=1 git@github.com:devreal/ompi.git openmpi
cd openmpi
git submodule update --init --recursive
./autogen.pl
mkdir -p build
cd build
../configure --prefix="$(realpath "../install")"
make -j8
make install