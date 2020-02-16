# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1._util
import awkward1._numpy
import awkward1.layout
import awkward1.operations.convert

def count(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.size(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.count(axis=axis, mask=False, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.count_nonzero)
def count_nonzero(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.count_nonzero(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.count_nonzero(axis=axis, mask=False, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.sum)
def sum(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.sum(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.sum(axis=axis, mask=False, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.prod)
def prod(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] * reduce(xs[1:])
        return reduce([numpy.prod(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.prod(axis=axis, mask=False, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.any)
def any(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] or reduce(xs[1:])
        return reduce([numpy.any(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.any(axis=axis, mask=False, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.all)
def all(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] and reduce(xs[1:])
        return reduce([numpy.all(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.all(axis=axis, mask=False, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.min)
def min(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 0:
                return None
            elif len(xs) == 1:
                return xs[0]
            else:
                x, y = xs[0], reduce(xs[1:])
                return x if x < y else y
        tmp = awkward1._util.completely_flatten(layout)
        return reduce([numpy.min(x) for x in tmp if len(x) > 0])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.min(axis=axis, mask=True, keepdims=keepdims), behavior)

@awkward1._numpy.implements(numpy.max)
def max(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 0:
                return None
            elif len(xs) == 1:
                return xs[0]
            else:
                x, y = xs[0], reduce(xs[1:])
                return x if x > y else y
        tmp = awkward1._util.completely_flatten(layout)
        return reduce([numpy.max(x) for x in tmp if len(x) > 0])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.max(axis=axis, mask=True, keepdims=keepdims), behavior)

### The following are not strictly reducers, but are defined in terms of reducers and ufuncs.

def moment(x, n, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        if weight is None:
            sumw   = count(x, axis=axis, keepdims=keepdims)
            sumwxn = sum(x**n, axis=axis, keepdims=keepdims)
        else:
            sumw   = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwxn = sum((x*weight)**n, axis=axis, keepdims=keepdims)
        return numpy.true_divide(sumwxn, sumw)

@awkward1._numpy.implements(numpy.mean)
def mean(x, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        if weight is None:
            sumw  = count(x, axis=axis, keepdims=keepdims)
            sumwx = sum(x, axis=axis, keepdims=keepdims)
        else:
            sumw  = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwx = sum(x*weight, axis=axis, keepdims=keepdims)
        return numpy.true_divide(sumwx, sumw)

@awkward1._numpy.implements(numpy.var)
def var(x, weight=None, ddof=0, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        xmean = mean(x, weight=weight, axis=axis, keepdims=keepdims)
        if weight is None:
            sumw   = count(x, axis=axis, keepdims=keepdims)
            sumwxx = sum((x - xmean)**2, axis=axis, keepdims=keepdims)
        else:
            sumw   = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwxx = sum((x - xmean)**2 * weight, axis=axis, keepdims=keepdims)
        if ddof != 0:
            return numpy.true_divide(sumwxx, sumw) * numpy.true_divide(sumw, sumw - ddof)
        else:
            return numpy.true_divide(sumwxx, sumw)

@awkward1._numpy.implements(numpy.std)
def std(x, weight=None, ddof=0, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        return numpy.sqrt(var(x, weight=weight, ddof=ddof, axis=axis, keepdims=keepdims))

def covar(x, y, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        xmean = mean(x, weight=weight, axis=axis, keepdims=keepdims)
        ymean = mean(y, weight=weight, axis=axis, keepdims=keepdims)
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims)
            sumwxy = sum((x - xmean)*(y - ymean), axis=axis, keepdims=keepdims)
        else:
            sumw = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwxy = sum((x - xmean)*(y - ymean)*weight, axis=axis, keepdims=keepdims)
        return numpy.true_divide(sumwxy, sumw)

def corr(x, y, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        xmean = mean(x, weight=weight, axis=axis, keepdims=keepdims)
        ymean = mean(y, weight=weight, axis=axis, keepdims=keepdims)
        xdiff = x - xmean
        ydiff = y - ymean
        if weight is None:
            sumwxx = sum(xdiff**2, axis=axis, keepdims=keepdims)
            sumwyy = sum(ydiff**2, axis=axis, keepdims=keepdims)
            sumwxy = sum(xdiff*ydiff, axis=axis, keepdims=keepdims)
        else:
            sumwxx = sum((xdiff**2)*weight, axis=axis, keepdims=keepdims)
            sumwyy = sum((ydiff**2)*weight, axis=axis, keepdims=keepdims)
            sumwxy = sum((xdiff*ydiff)*weight, axis=axis, keepdims=keepdims)
        return numpy.true_divide(sumwxy, numpy.sqrt(sumwxx * sumwyy))

def linearfit(x, y, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims)
            sumwx  = sum(x, axis=axis, keepdims=keepdims)
            sumwy  = sum(y, axis=axis, keepdims=keepdims)
            sumwxx = sum(x**2, axis=axis, keepdims=keepdims)
            sumwxy = sum(x*y, axis=axis, keepdims=keepdims)
        else:
            sumw = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwx  = sum(x*weight, axis=axis, keepdims=keepdims)
            sumwy  = sum(y*weight, axis=axis, keepdims=keepdims)
            sumwxx = sum((x**2)*weight, axis=axis, keepdims=keepdims)
            sumwxy = sum(x*y*weight, axis=axis, keepdims=keepdims)
        delta           = (sumw*sumwxx) - (sumwx*sumwx)
        intercept       = numpy.true_divide(((sumwxx*sumwy) - (sumwx*sumwxy)), delta)
        slope           = numpy.true_divide(((sumw*sumwxy) - (sumwx*sumwy)), delta)
        intercept_error = numpy.sqrt(numpy.true_divide(sumwxx, delta))
        slope_error     = numpy.sqrt(numpy.true_divide(sumw, delta))

        intercept       = awkward1.operations.convert.tolayout(intercept, allowrecord=True, allowother=True)
        slope           = awkward1.operations.convert.tolayout(slope, allowrecord=True, allowother=True)
        intercept_error = awkward1.operations.convert.tolayout(intercept_error, allowrecord=True, allowother=True)
        slope_error     = awkward1.operations.convert.tolayout(slope_error, allowrecord=True, allowother=True)

        scalar = not isinstance(intercept, awkward1.layout.Content) and not isinstance(slope, awkward1.layout.Content) and not isinstance(intercept_error, awkward1.layout.Content) and not isinstance(slope_error, awkward1.layout.Content)

        if not isinstance(intercept, (awkward1.layout.Content, awkward1.layout.Record)):
            intercept = awkward1.layout.NumpyArray(numpy.array([intercept]))
        if not isinstance(slope, (awkward1.layout.Content, awkward1.layout.Record)):
            slope = awkward1.layout.NumpyArray(numpy.array([slope]))
        if not isinstance(intercept_error, (awkward1.layout.Content, awkward1.layout.Record)):
            intercept_error = awkward1.layout.NumpyArray(numpy.array([intercept_error]))
        if not isinstance(slope_error, (awkward1.layout.Content, awkward1.layout.Record)):
            slope_error = awkward1.layout.NumpyArray(numpy.array([slope_error]))

        out = awkward1.layout.RecordArray([intercept, slope, intercept_error, slope_error], ["intercept", "slope", "intercept_error", "slope_error"])
        out.setparameter("__record__", "LinearFit")
        if scalar:
            out = out[0]

        return awkward1._util.wrap(out, awkward1._util.behaviorof(x, y))

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("collections", "numpy", "awkward1")]
