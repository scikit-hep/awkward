#!/usr/bin/env bash
# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

set -e

AWKWARD_VERSION=`cat VERSION_INFO`
# CUDA_VERSION=`nvcc --version | grep -ho "release [0-9\.]*" | sed 's/release //'`
# RUNTIME_CUDA_VERSION=`nvidia-smi | grep -ho "CUDA Version: [0-9\.]*" | sed 's/CUDA Version: //'`

rm -rf build dist
mkdir build
cp -r src/awkward1_cuda_kernels build

docker run -it -v`pwd`:/home -w/home docker.io/nvidia/cuda:10.1-devel-ubuntu18.04 nvcc -Xcompiler -fPIC -Iinclude src/cuda-kernels/*.cu --shared -o build/awkward1_cuda_kernels/libawkward1-cuda-kernels.so

cat > build/cuda-setup.py << EOF

import setuptools
from setuptools import setup

setup(name = "awkward1-cuda-kernels",
      packages = ["awkward1_cuda_kernels"],
      package_dir = {"": "build"},
      package_data = {"awkward1_cuda_kernels": ["*.so"]},
      version = open("VERSION_INFO").read().strip(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "Development of awkward 1.0, to replace scikit-hep/awkward-array in 2020.",
      long_description = "",
      url = "https://github.com/scikit-hep/awkward-1.0",
      download_url = "https://github.com/scikit-hep/awkward-1.0/releases",
      license = "BSD 3-clause",
      test_suite = "tests-cuda",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
      install_requires = ["cupy"],
      tests_require = ["pytest"],
      classifiers = [
#         "Development Status :: 1 - Planning",
#         "Development Status :: 2 - Pre-Alpha",
          "Development Status :: 3 - Alpha",
#         "Development Status :: 4 - Beta",
#         "Development Status :: 5 - Production/Stable",
#         "Development Status :: 6 - Mature",
#         "Development Status :: 7 - Inactive",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ])

EOF

python build/cuda-setup.py bdist_wheel
pip install dist/awkward1_cuda_kernels-$AWKWARD_VERSION-py3-none-any.whl
