# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

autograd = pytest.importorskip("autograd")

def tanh(x):
    y = autograd.numpy.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

def test():
    grad_tanh = awkward1.autograd(tanh)

    xs = awkward1.Array(numpy.linspace(-3, 3, 10))
    assert awkward1.tolist(xs) == pytest.approx([-3.0, -2.3333333333333335, -1.6666666666666667, -1.0, -0.3333333333333335, 0.33333333333333304, 1.0, 1.666666666666666, 2.333333333333333, 3.0])

    assert awkward1.tolist(tanh(xs)) == pytest.approx([-0.9950547536867305, -0.9813680813098666, -0.9311096086675776, -0.7615941559557649, -0.32151273753163445, 0.32151273753163406, 0.7615941559557649, 0.9311096086675775, 0.9813680813098666, 0.9950547536867306])

    assert awkward1.tolist(grad_tanh(xs)) == pytest.approx([0.009866037165439843, 0.036916688986191167, 0.1330348966469106, 0.4199743416140259, 0.8966295596049142, 0.8966295596049146, 0.419974341614026, 0.13303489664691054, 0.03691668898619103, 0.009866037165440192])
