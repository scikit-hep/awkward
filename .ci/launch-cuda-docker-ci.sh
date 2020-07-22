#!/bin/bash

docker build -f ../dev/docker/docker_cuda102/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.2 ../.
docker run -it --gpus all awkward1-cuda-tests:1.0-cuda10.2 sh .ci/run-cuda-tests.sh trickarcher/python_cuda_interface

docker build -f ../dev/docker/docker_cuda100/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.0 ../.
docker run -it --gpus all awkward1-cuda-tests:1.0-cuda10.0 sh .ci/run-cuda-tests.sh trickarcher/python_cuda_interface

docker build -f ../dev/docker/docker_cuda900/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda9.0 ../.
docker run -it --gpus all awkward1-cuda-tests:1.0-cuda9.0 sh .ci/run-cuda-tests.sh trickarcher/python_cuda_interface
