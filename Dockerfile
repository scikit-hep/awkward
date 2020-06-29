FROM nvidia/cuda:10.2-devel
FROM python:3

WORKDIR /awkward

COPY . ./

RUN nvcc

RUN cd cuda-kernels && rm -rf include src VERSION_INFO dist awkward1_cuda_kernels.egg-info
RUN cd cuda-kernels && python setup.py sdist
RUN cd cuda-kernels && pip install -v .