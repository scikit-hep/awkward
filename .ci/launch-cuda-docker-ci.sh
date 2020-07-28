#!/bin/bash

LOGS=false
GIT_BRANCH=master

while getopts ":b:l" opt; do
  case $opt in
    b)
      GIT_BRANCH=$OPTARG
      ;;
    l)
      LOGS=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

LOG_FOLDER=$( date +%F_%H-%M )

if [ `git ls-remote --heads https://github.com/scikit-hep/awkward-1.0.git $GIT_BRANCH | wc -l` -eq 0 ]
then
    echo "Branch does not exist"
    exit
fi

if $LOGS 
then
	mkdir /home/ubuntu/awkward-1.0/.ci/logs/$LOG_FOLDER

	docker build -f /home/ubuntu/awkward-1.0/dev/docker/docker_cuda102/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.2 /home/ubuntu/awkward-1.0/.
	docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda10.2 sh .ci/run-cuda-tests.sh -b $GIT_BRANCH -l $LOG_FOLDER -c 102

	docker build -f /home/ubuntu/awkward-1.0/dev/docker/docker_cuda100/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.0 /home/ubuntu/awkward-1.0/.
	docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda10.0 sh .ci/run-cuda-tests.sh -b $GIT_BRANCH -l $LOG_FOLDER -c 100

	docker build -f /home/ubuntu/awkward-1.0/dev/docker/docker_cuda900/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda9.0 /home/ubuntu/awkward-1.0/.
	docker run -it --gpus all -v /home/ubuntu/awkward-1.0/.ci/logs:/awkward-1.0/.ci/logs awkward1-cuda-tests:1.0-cuda9.0 sh .ci/run-cuda-tests.sh -b $GIT_BRANCH -l $LOG_FOLDER -c 900
else
	docker build -f /home/ubuntu/awkward-1.0/dev/docker/docker_cuda102/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.2 /home/ubuntu/awkward-1.0/.
	docker run -it --gpus all awkward1-cuda-tests:1.0-cuda10.2 sh .ci/run-cuda-tests.sh -b $GIT_BRANCH

	docker build -f /home/ubuntu/awkward-1.0/dev/docker/docker_cuda100/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda10.0 /home/ubuntu/awkward-1.0/.
	docker run -it --gpus all awkward1-cuda-tests:1.0-cuda10.0 sh .ci/run-cuda-tests.sh -b $GIT_BRANCH 

	docker build -f /home/ubuntu/awkward-1.0/dev/docker/docker_cuda900/Dockerfile.build -t awkward1-cuda-tests:1.0-cuda9.0 /home/ubuntu/awkward-1.0/.
	docker run -it --gpus all awkward1-cuda-tests:1.0-cuda9.0 sh .ci/run-cuda-tests.sh -b $GIT_BRANCH
fi	
