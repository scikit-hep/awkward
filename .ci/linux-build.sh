#!/bin/bash

cd /io/

/opt/python/cp37-cp37m/bin/pip install cmake==3.13.3
export PATH=/opt/_internal/cpython-3.7.4/lib/python3.7/site-packages/cmake/data/bin:$PATH

for PYBIN in /opt/python/*/bin; do
    echo "========================================================="
    echo $PYBIN
    echo $PATH
    which cmake
    echo "========================================================="
    "${PYBIN}/python" setup.py bdist_wheel -p $PLAT
done
