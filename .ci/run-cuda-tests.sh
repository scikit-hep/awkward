#!/bin/bash
########################## do-tests.sh

set -e

conda --version
python --version

git checkout $1
git pull
git status

cp -a include src VERSION_INFO cuda-kernels

cd cuda-kernels
pip install .
cd ..

python -c 'import awkward1_cuda_kernels; print(awkward1_cuda_kernels.shared_library_path)'

python localbuild.py --pytest tests

python -m pytest -vvrs tests-cuda
