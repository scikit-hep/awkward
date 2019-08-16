#!/bin/bash

cd /io/

for PYBIN in /opt/python/*/bin; do
    echo "========================================================="
    echo $PYBIN
    echo "========================================================="
    "${PYBIN}/python" setup.py bdist_wheel -p $PLAT
done
