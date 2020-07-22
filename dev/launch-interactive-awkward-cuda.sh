#!/bin/bash

docker run -it --gpus all awkward1-cuda-tests:1.0-cuda10.2 sh do-tests.sh trickarcher/python_cuda_interface
