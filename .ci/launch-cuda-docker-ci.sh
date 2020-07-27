#!/bin/bash

docker build -f ../dev/docker/docker_cuda102/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.2 ../.
docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda10.2 sh .ci/run-cuda-tests.sh $1 102

docker build -f ../dev/docker/docker_cuda100/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.0 ../.
docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda10.0 sh .ci/run-cuda-tests.sh $1 100

docker build -f ../dev/docker/docker_cuda900/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda9.0 ../.
docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda9.0 sh .ci/run-cuda-tests.sh $1 900 
