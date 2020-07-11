FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /awkward

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y \
	    curl \
	    git \
	    cmake \	
	    python3-pip \
	    python3-distutils
RUN nvcc --version
COPY . .


RUN ln -s /usr/bin/python3 /usr/bin/python && \
ln -s /usr/bin/pip3 /usr/bin/pip && \
pip install --upgrade pip && \
pip install setuptools && \
pip install -r requirements.txt -r requirements-test.txt && \
pip install -v .  && \
cd cuda-kernels && \
python setup.py sdist && \
pip install -v . && \
cd .. && \
pytest tests_cuda

