#!/bin/bash

cd /io/

if [[ $PLAT != "manylinux2010_x86_64" ]]; then
    /opt/python/cp37-cp37m/bin/pip install cmake
    export PATH=/opt/_internal/cpython-3.7.4/lib/python3.7/site-packages/cmake/data/bin:$PATH
fi

for PYBIN in /opt/python/*/bin; do
    echo "========================================================="
    echo $PYBIN
    echo "========================================================="
    "${PYBIN}/python" setup.py bdist_wheel -p $PLAT
done
