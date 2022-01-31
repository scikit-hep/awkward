#!/usr/bin/env bash
# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

set -e

PLATFORM="manylinux2014_x86_64"
install_flag=false

help()
{
  echo ""
  echo "Usage: $0 -c <cuda-version> -i"
  echo -e "\t-c The CUDA Version against which to build the library"
  echo -e "\t-i Install the awkward_cuda_kernels library"
  exit 1 # Exit script after printing help
}

while getopts "c:i" opt
do
  case "$opt" in
    c) CUDA_VERSION="$OPTARG" ;;
    i) install_flag=true ;;
    ?) help ;;
  esac
done

CUPY_CUDA_VERSION=${CUDA_VERSION//\./}
AWKWARD_VERSION=$(tr -d '[:space:]' < VERSION_INFO)

# The choices in the CUDA Version is strictly dependent on Cupy.
case $CUDA_VERSION in
  "11.5"* | "11.4"* | "11.3"* | "11.2"* | "11.1"* | "11.0"*)
    CUPY_CUDA_VERSION=${CUPY_CUDA_VERSION:0:3}
    echo $CUPY_CUDA_VERSION
    export DOCKER_IMAGE_TAG="$CUDA_VERSION-devel-ubuntu20.04"
    ;;
  "10.2")
    export DOCKER_IMAGE_TAG="$CUDA_VERSION-devel-ubuntu18.04"
    ;;
  *)
    echo "Docker image for CUDA version $CUDA_VERSION is not known"
    exit 1
    ;;
esac

rm -rf build dist
mkdir build
cp -r src/awkward_cuda_kernels build

{
  echo "__version__ = \"$AWKWARD_VERSION\""
  echo "cuda_version = \"$CUDA_VERSION\""
  echo "docker_image = \"docker.io/nvidia/cuda:$DOCKER_IMAGE_TAG\""
} >> build/awkward_cuda_kernels/__init__.py

export DOCKER_ARGS=("-v$PWD:/home" -w/home "docker.io/nvidia/cuda:$DOCKER_IMAGE_TAG")
export BUILD_SHARED_LIBRARY=(nvcc -std=c++11 -Xcompiler -fPIC -Xcompiler "-DVERSION_INFO=$CUDA_VERSION" -Iinclude src/cuda-kernels/*.cu --shared -o build/awkward_cuda_kernels/libawkward-cuda-kernels.so)

docker run "${DOCKER_ARGS[@]}" "${BUILD_SHARED_LIBRARY[@]}"

cat > build/cuda-setup.py << EOF

import setuptools
from setuptools import setup
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name = "awkward-cuda-kernels",
      packages = ["awkward_cuda_kernels"],
      package_dir = {"": "build"},
      package_data = {"awkward_cuda_kernels": ["*.so"]},
      distclass = BinaryDistribution,
      version = "$CUDA_VERSION",
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "CUDA plug-in for Awkward Array, enables GPU-bound arrays and operations.",
      long_description = "This plug-in is experimental. Instructions on how to use it will be provided with its first stable release.",
      long_description_content_type = "text/markdown",
      url = "https://github.com/scikit-hep/awkward-1.0",
      download_url = "https://github.com/scikit-hep/awkward-1.0/releases",
      license = "BSD 3-clause",
      test_suite = "tests-cuda",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
      install_requires = ["cupy_cuda$CUPY_CUDA_VERSION"],
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
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
#         "Programming Language :: Python :: 2.7",
#         "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ])

EOF

python build/cuda-setup.py bdist_wheel --plat-name manylinux2014_x86_64

if [ "$install_flag" == "true" ]; then
    pip install dist/awkward_cuda_kernels-*.whl
fi
