# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1.layout
import awkward1.operations.convert
import awkward1._numpy
import awkward1._util

def elementwise_grad(fun, argnum=0, *nary_op_args, **nary_op_kwargs):
    import autograd
    gradfun = autograd.elementwise_grad(fun, argnum, *nary_op_args, **nary_op_kwargs)

    def broadcast(*args, **kwargs):
        nextargs = [awkward1.operations.convert.tolayout(x, allowrecord=True, allowother=True) for x in args]

        def getfunction(inputs):
            if all(isinstance(x, awkward1.layout.NumpyArray) or not isinstance(x, awkward1.layout.Content) for x in inputs):
                arrays = [numpy.asarray(x) if isinstance(x, awkward1.layout.NumpyArray) else x for x in inputs]
                return lambda depth: (awkward1.layout.NumpyArray(gradfun(*arrays)),)

            else:
                return None

        out = awkward1._util.broadcast_and_apply(nextargs, getfunction)
        assert isinstance(out, tuple) and len(out) == 1
        return awkward1._util.wrap(out[0], awkward1._util.behaviorof(args))

    return broadcast

elementwise_grad.elementwise_grad = elementwise_grad
