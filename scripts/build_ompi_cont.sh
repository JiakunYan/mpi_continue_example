#!/usr/bin/env bash

set -e

mkdir -p build
cd build
if [ ! -d ./openmpi ]; then
  git clone --branch=mpi-continue-master --depth=1 git@github.com:devreal/ompi.git openmpi
fi
cd openmpi
git submodule update --init --recursive
./autogen.pl
mkdir -p build
cd build
../configure --prefix="$(realpath "../install")"
make install -j8