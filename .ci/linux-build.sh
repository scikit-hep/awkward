#!/bin/bash

cd /io/

/opt/python/cp37-cp37m/bin/pip install cmake==3.13.3

echo `ls -d /opt/_internal/cpython-3.7.*/lib/python3.7/site-packages/cmake/data/bin`
ls /opt/_internal/cpython-3.7.*/lib/python3.7/site-packages/cmake/data/bin

export PATH=`ls -d /opt/_internal/cpython-3.7.*/lib/python3.7/site-packages/cmake/data/bin`:$PATH

echo "PATH:" $PATH
echo "which cmake:" `which cmake`

for PYBIN in /opt/python/*/bin; do
    echo "========================================================="
    echo $PYBIN
    echo "========================================================="
    "${PYBIN}/pip" wheel /io/ -v -w wheelhouse/
done

# Bundle external shared libraries into the wheels
# And make sure they are manylinux ready
for whl in wheelhouse/awkward*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

