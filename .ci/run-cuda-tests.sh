#!/bin/bash
########################## do-tests.sh

set -e
LOGS=false
GIT_BRANCH=master
CUDA_VERSION=102
LOG_FOLDER=".ci/logs/"

while getopts ":b:l:c:" opt; do
  case $opt in
    b)
			GIT_BRANCH=$OPTARG
			if [ `git ls-remote --heads https://github.com/scikit-hep/awkward-1.0.git $GIT_BRANCH | wc -l` -eq 0 ]
			then
			    echo "Branch does not exist"
			    exit
			fi
			;;
    l)
      LOGS=true
      LOG_FOLDER=".ci/logs/"$OPTARG
      ;;
    c)
			CUDA_VERSION=$OPTARG
			;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

LOG_FILE_AWKWARD1BUILD=$CUDA_VERSION"awkward1-build-test"
LOG_FILE_AWKWARD1CUDABUILD=$CUDA_VERSION"awkward1-cuda-build"
LOG_FILE_AWKWARD1CUDATEST=$CUDA_VERSION"awkward1-cuda-test"

conda --version
python --version

git checkout $GIT_BRANCH
git pull
git status
git submodule update --init

cp -a include src VERSION_INFO cuda-kernels

if $LOGS
then
	cd cuda-kernels
	pip install . | tee ../${LOG_FOLDER}/${LOG_FILE_AWKWARD1CUDABUILD} 
	cd ..

	python -c 'import awkward1_cuda_kernels; print(awkward1_cuda_kernels.shared_library_path)'

	python localbuild.py --pytest tests | tee ${LOG_FOLDER}/${LOG_FILE_AWKWARD1BUILD}
	python -m pytest -vvrs tests-cuda | tee ${LOG_FOLDER}/${LOG_FILE_AWKWARD1CUDATEST}
else
	cd cuda-kernels
	pip install .
	cd ..

	python -c 'import awkward1_cuda_kernels; print(awkward1_cuda_kernels.shared_library_path)'

	python localbuild.py --pytest tests
	python -m pytest -vvrs tests-cuda
fi

