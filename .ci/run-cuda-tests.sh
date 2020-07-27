#!/bin/bash
########################## do-tests.sh

set -e
LOG_FILE_AWKWARD1BUILD="awkward1-cuda-tests "$1" awkward1-build-test"
LOG_FILE_AWKWARD1CUDABUILD="awkward1-cuda-tests "$1" awkward1-cuda-build"
LOG_FILE_AWKWARD1CUDATEST="awkward1-cuda-tests "$1" awkward1-cuda-test"
LOG_FOLDER=".ci/logs/"$( date )
mkdir LOG_FOLDER
ls
conda --version
python --version

git checkout $1
git pull
git status
git submodule update --init

cp -a include src VERSION_INFO cuda-kernels

cd cuda-kernels
pip install . | tee ../"${LOG_FOLDER}"/"${LOG_FILE_AWKWARD1CUDABUILD}"
cd ..

python -c 'import awkward1_cuda_kernels; print(awkward1_cuda_kernels.shared_library_path)'

python localbuild.py --pytest tests | tee "${LOG_FOLDER}"/"${LOG_FILE_AWKWARD1BUILD}"

python -m pytest -vvrs tests-cuda | tee
"${LOG_FOLDER}"/"${LOG_FILE_AWKWARD1CUDATEST}"
