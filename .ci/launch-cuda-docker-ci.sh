#!/bin/bash

LOG_FOLDER=$( date +%F_%H-%M )

mkdir logs/$LOG_FOLDER

docker build -f ../dev/docker/docker_cuda102/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.2 ../.
docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda10.2 sh .ci/run-cuda-tests.sh $1 102 $LOG_FOLDER

docker build -f ../dev/docker/docker_cuda100/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.0 ../.
docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda10.0 sh .ci/run-cuda-tests.sh $1 100 $LOG_FOLDER

docker build -f ../dev/docker/docker_cuda900/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda9.0 ../.
docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda9.0 sh .ci/run-cuda-tests.sh $1 900  $LOG_FOLDER
