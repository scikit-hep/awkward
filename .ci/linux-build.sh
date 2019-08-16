#!/bin/bash

cd /io/awkward-1.0/

for PYBIN in /opt/python/*/bin; do
    echo "========================================================="
    echo $PYBIN
    echo "========================================================="
    "${PYBIN}/python" setup.py bdist_wheel -p $PLAT
    mv dist/* /io/wheelhouse/
done
