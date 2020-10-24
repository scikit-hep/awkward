#!/usr/bin/env bash

echo "Copying files"

cp standard_parallel_algorithms.h loop-dependent-variable-kernels/manual_*.cu ../../src/cuda-kernels/
cd ../..

echo "Running setup"
sudo pip uninstall awkward1_cuda_kernels && sudo ./cuda-build.sh --install
python python dev/generate-tests.py

echo "Testing"
pytest -vvrs tests-cuda-kernels

echo "Cleaning up"
mv src/cuda-kernels/manual_awkward_ListArray_num.cu src/

rm src/cuda-kernels/manual* src/cuda-kernels/standard_parallel_algorithms.h

mv src/manual_awkward_ListArray_num.cu src/cuda-kernels/
