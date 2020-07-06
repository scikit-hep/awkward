FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /awkward

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y \
	    curl \
	    git \
	    cmake \	
	    python \
	    python-pip \
	    python3-distutils
RUN nvcc --version
COPY . .
RUN ls
RUN pip install setuptools
RUN pip install -r requirements.txt
RUN pip install -r requirements-test.txt
RUN python localbuild.py --pytest tests
RUN mkdir  cuda-kernels/build
RUN cd cuda-kernels/build && cmake ..
