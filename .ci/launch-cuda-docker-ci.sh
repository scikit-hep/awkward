#!/bin/bash

NVIDIA_DRIVER_VER=$(modinfo nvidia --field version)

if [ $(echo "$NVIDIA_DRIVER_VER > 440" |bc -l) ]
then
	docker build -f docker/docker_cuda102/Dockerfile.build -t awkward1_cuda:10.2 ../.
	docker build -f docker/docker_cuda102/Dockerfile.test -t awkward1_cuda_test:10.2 ../.
	docker run -it --gpus all awkward1_cuda_test:10.2 

	docker build -f docker/docker_cuda100/Dockerfile.build -t awkward1_cuda:10.0 ../.
	docker build -f docker/docker_cuda100/Dockerfile.test -t awkward1_cuda_test:10.0 ../.
	docker run -it --gpus all awkward1_cuda_test:10.0 

	docker build -f docker/docker_cuda900/Dockerfile.build -t awkward1_cuda:9.0 ../.
	docker build -f docker/docker_cuda900/Dockerfile.test -t awkward1_cuda_test:9.0 ../.
	docker run -it --gpus all awkward1_cuda_test:9.0 

elif [ $(echo "$NVIDIA_DRIVER_VER > 410.48" |bc -l) ]
then 
	docker build -f docker/docker_cuda100/Dockerfile.build -t awkward1_cuda:10.0 ../.
	docker build -f docker/docker_cuda100/Dockerfile.test -t awkward1_cuda_test:10.0 ../.
	docker run -it --gpus all awkward1_cuda_test:10.0 

	docker build -f docker/docker_cuda900/Dockerfile.build -t awkward1_cuda:9.0 ../.
	docker build -f docker/docker_cuda900/Dockerfile.test -t awkward1_cuda_test:9.0 ../.
	docker run -it --gpus all awkward1_cuda_test:9.0 

elif [ $(echo "$NVIDIA_DRIVER_VER > 384.81" |bc -l) ]
then 
	docker build -f docker/docker_cuda900/Dockerfile.build -t awkward1_cuda:9.0 ../.
	docker build -f docker/docker_cuda900/Dockerfile.test -t awkward1_cuda_test:9.0 ../.
	docker run -it --gpus all awkward1_cuda_test:9.0 
else
	echo "NVIDIA Driver version greater than 384.81 required!"
fi
