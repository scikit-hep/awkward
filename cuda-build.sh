#!/usr/bin/env bash
# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

set -e

PLATFORM="manylinux2014_x86_64"
CUDA_VERSION=$1
CUPY_CUDA_VERSION=${CUDA_VERSION//\./}

AWKWARD_VERSION=$(tr -d '[:space:]' < VERSION_INFO)

case $CUDA_VERSION in
  "11.5.1")
    CUPY_CUDA_VERSION=11.5
    export DOCKER_IMAGE_TAG="$CUDA_VERSION-devel-ubuntu20.04"
    ;;
  "11.0" | "10.2" | "10.1" | "10.0" | "9.2")
    export DOCKER_IMAGE_TAG="$CUDA_VERSION-devel-ubuntu18.04"
    ;;
  "9.1" | "9.0" | "8.0" )
    export DOCKER_IMAGE_TAG="$CUDA_VERSION-devel-ubuntu16.04"
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
export BUILD_SHARED_LIBRARY=(nvcc -std=c++11 -Xcompiler -fPIC -Xcompiler "-DVERSION_INFO=$AWKWARD_VERSION" -Iinclude src/cuda-kernels/*.cu --shared -o build/awkward_cuda_kernels/libawkward-cuda-kernels.so)

docker run "${DOCKER_ARGS[@]}" "${BUILD_SHARED_LIBRARY[@]}"

cat > build/cuda-setup.py << EOF
import setuptools
from setuptools import setup
setup(name = "awkward-cuda-kernels",
      packages = ["awkward_cuda_kernels"],
      package_dir = {"": "build"},
      package_data = {"awkward_cuda_kernels": ["*.so"]},
      version = open("VERSION_INFO").read().strip(),
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

python build/cuda-setup.py bdist_wheel

cd dist
rm -f "awkward_cuda_kernels-$AWKWARD_VERSION-py3-none-$PLATFORM.whl"

unzip "awkward_cuda_kernels-$AWKWARD_VERSION-py3-none-any.whl"

cp "awkward_cuda_kernels-$AWKWARD_VERSION.dist-info/WHEEL" tmp_WHEEL
sed "s/Root-Is-Purelib: true/Root-Is-Purelib: false/" < tmp_WHEEL | sed "s/Tag: py3-none-any/Tag: py3-none-$PLATFORM/" > "awkward_cuda_kernels-$AWKWARD_VERSION.dist-info/WHEEL"

zip "awkward_cuda_kernels-$AWKWARD_VERSION-py3-none-$PLATFORM.whl" -r awkward_cuda_kernels "awkward_cuda_kernels-$AWKWARD_VERSION.dist-info"

cd ..

if [ "$1" == "--install" ]; then
    pip install "dist/awkward_cuda_kernels-$AWKWARD_VERSION-py3-none-$PLATFORM.whl"
fi
