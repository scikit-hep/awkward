# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1._util
import awkward1._connect._numpy
import awkward1.layout
import awkward1.operations.convert

def count(array, axis=None, keepdims=False, maskidentity=False):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.size(x)
                         for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.count(axis=axis,
                                                mask=maskidentity,
                                                keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.count_nonzero)
def count_nonzero(array, axis=None, keepdims=False, maskidentity=False):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.count_nonzero(x)
                         for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.count_nonzero(axis=axis,
                                                        mask=maskidentity,
                                                        keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.sum)
def sum(array, axis=None, keepdims=False, maskidentity=False):
    """
    Args:
        array: Data to sum over.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer descreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        maskidentity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Sums over `array` (many types supported, including all Awkward Arrays
    and Records). The identity of addition is `0` and it is usually not
    masked. This operation is the same as NumPy's
    [sum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, consider this `array`, in which all lists at a given dimension
    have the same length.

        ak.Array([[ 0.1,  0.2,  0.3],
                  [10.1, 10.2, 10.3],
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2, 30.3]])

    A sum over `axis=-1` combines the inner lists, leaving one value per
    outer list:

        >>> ak.sum(array, axis=-1)
        <Array [0.6, 30.6, 60.6, 90.6] type='4 * float64'>

    while a sum over `axis=0` combines the outer lists, leaving one value
    per inner list:

        >>> ak.sum(array, axis=0)
        <Array [60.4, 60.8, 61.2] type='3 * float64'>

    Now with some values missing,

        ak.Array([[ 0.1,  0.2      ],
                  [10.1            ],
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2      ]])

    The sum over `axis=-1` results in

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>

    and the sum over `axis=0` results in

        >>> ak.sum(array, axis=0)
        <Array [60.4, 50.6, 20.3] type='3 * float64'>

    How we ought to sum over the innermost lists is unambiguous, but for all
    other `axis` values, we must choose whether to align contents to the
    left before summing, to the right before summing, or something else.
    As suggested by the way the text has been aligned, we choose the
    left-alignment convention: the first `axis=0` result is the sum of all
    first elements

        60.4 = 0.1 + 10.1 + 20.1 + 30.1

    the second is the sum of all second elements

        50.6 = 0.2 + 20.2 + 30.2

    and the third is the sum of the only third element

        20.3 = 20.3

    The same is true if the values were None, rather than gaps:

        ak.Array([[ 0.1,  0.2, None],
                  [10.1, None, None],
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2, None]])

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>
        >>> ak.sum(array, axis=0)
        <Array [60.4, 50.6, 20.3] type='3 * float64'>

    However, the missing value placeholder, None, allows us to align the
    remaining data differently:

        ak.Array([[None,  0.1,  0.2],
                  [None, None, 10.1],
                  [20.1, 20.2, 20.3],
                  [None, 30.1, 30.2]])

    Now the `axis=-1` result is the same but the `axis=0` result has changed:

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>
        >>> ak.sum(array, axis=0)
        <Array [20.1, 50.4, 60.8] type='3 * float64'>

    because

        20.1 = 20.1
        50.4 = 0.1 + 20.2 + 30.1
        60.8 = 0.2 + 10.1 + 20.3 + 30.2

    If, instead of missing numbers, we had missing lists,

        ak.Array([[ 0.1,  0.2,  0.3],
                  None,
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2, 30.3]])

    then the placeholder would pass through the `axis=-1` sum because summing
    over the inner dimension shouldn't change the length of the outer
    dimension.

        >>> ak.sum(array, axis=-1)
        <Array [0.6, None, 60.6, 90.6] type='4 * ?float64'>

    However, the `axis=0` sum loses information about the None value.

        >>> ak.sum(array, axis=0)
        <Array [50.3, 50.6, 50.9] type='3 * float64'>

    which is

        50.3 = 0.1 + (None) + 20.1 + 30.1
        50.6 = 0.2 + (None) + 20.2 + 30.2
        50.9 = 0.3 + (None) + 20.3 + 30.3

    An `axis=0` sum would be reducing that information if it had not been
    None, anyway. If the None values were replaced with `0`, the result for
    `axis=0` would be the same. The result for `axis=-1` would not be the
    same because this None is in the `0` axis, not the axis that `axis=-1`
    sums over.

    The `keepdims` parameter ensures that the number of dimensions does not
    change: scalar results are put into new length-1 dimensions:

        >>> ak.sum(array, axis=-1, keepdims=True)
        <Array [[0.6], None, [60.6], [90.6]] type='4 * option[1 * float64]'>
        >>> ak.sum(array, axis=0, keepdims=True)
        <Array [[50.3, 50.6, 50.9]] type='1 * var * float64'>

    and `axis=None` ignores all None values and adds up everything in the
    array (`keepdims` has no effect).

        >>> ak.sum(array, axis=None)
        151.8

    The `maskidentity`, which has no equivalent in NumPy, inserts None in
    the output wherever a reduction takes place over zero elements. This is
    different from reductions that are otherwise equal to the identity or
    are equal to the identity by cancellation.

        >>> array = ak.Array([[2.2, 2.2], [4.4, -2.2, -2.2], [], [0.0]])
        >>> ak.sum(array, axis=-1)
        <Array [4.4, 0, 0, 0] type='4 * float64'>
        >>> ak.sum(array, axis=-1, maskidentity=True)
        <Array [4.4, 0, None, 0] type='4 * ?float64'>

    The third list is reduced to `0` if `maskidentity=False` because `0` is the
    identity of addition, but it is reduced to None if `maskidentity=True`.
    """
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.sum(x)
                         for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.sum(axis=axis,
                                              mask=maskidentity,
                                              keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.prod)
def prod(array, axis=None, keepdims=False, maskidentity=False):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] * reduce(xs[1:])
        return reduce([numpy.prod(x)
                         for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.prod(axis=axis,
                                               mask=maskidentity,
                                               keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.any)
def any(array, axis=None, keepdims=False, maskidentity=False):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] or reduce(xs[1:])
        return reduce([numpy.any(x)
                         for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.any(axis=axis,
                                              mask=maskidentity,
                                              keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.all)
def all(array, axis=None, keepdims=False, maskidentity=False):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] and reduce(xs[1:])
        return reduce([numpy.all(x)
                         for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.all(axis=axis,
                                              mask=maskidentity,
                                              keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.min)
def min(array, axis=None, keepdims=False, maskidentity=True):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
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
        return awkward1._util.wrap(layout.min(axis=axis,
                                              mask=maskidentity,
                                              keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.max)
def max(array, axis=None, keepdims=False, maskidentity=True):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
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
        return awkward1._util.wrap(layout.max(axis=axis,
                                              mask=maskidentity,
                                              keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.argmin)
def argmin(array, axis=None, keepdims=False, maskidentity=True):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        tmp = awkward1._util.completely_flatten(layout)
        return numpy.argmin(tmp, axis=None)
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.argmin(axis=axis,
                                                 mask=maskidentity,
                                                 keepdims=keepdims),
                                   behavior)

@awkward1._connect._numpy.implements(numpy.argmax)
def argmax(array, axis=None, keepdims=False, maskidentity=True):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if axis is None:
        tmp = awkward1._util.completely_flatten(layout)
        return numpy.argmax(tmp, axis=None)
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.argmax(axis=axis,
                                                 mask=maskidentity,
                                                 keepdims=keepdims),
                                   behavior)

# The following are not strictly reducers, but are defined in terms of
# reducers and ufuncs.

def moment(x, n, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        if weight is None:
            sumw   = count(x, axis=axis, keepdims=keepdims)
            sumwxn = sum(x**n, axis=axis, keepdims=keepdims)
        else:
            sumw   = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwxn = sum((x*weight)**n, axis=axis, keepdims=keepdims)
        return numpy.true_divide(sumwxn, sumw)

@awkward1._connect._numpy.implements(numpy.mean)
def mean(x, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        if weight is None:
            sumw  = count(x, axis=axis, keepdims=keepdims)
            sumwx = sum(x, axis=axis, keepdims=keepdims)
        else:
            sumw  = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwx = sum(x*weight, axis=axis, keepdims=keepdims)
        return numpy.true_divide(sumwx, sumw)

@awkward1._connect._numpy.implements(numpy.var)
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
            return (numpy.true_divide(sumwxx, sumw) *
                    numpy.true_divide(sumw, sumw - ddof))
        else:
            return numpy.true_divide(sumwxx, sumw)

@awkward1._connect._numpy.implements(numpy.std)
def std(x, weight=None, ddof=0, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        return numpy.sqrt(var(x,
                              weight=weight,
                              ddof=ddof,
                              axis=axis,
                              keepdims=keepdims))

def covar(x, y, weight=None, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        xmean = mean(x, weight=weight, axis=axis, keepdims=keepdims)
        ymean = mean(y, weight=weight, axis=axis, keepdims=keepdims)
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims)
            sumwxy = sum((x - xmean)*(y - ymean), axis=axis, keepdims=keepdims)
        else:
            sumw = sum(x*0 + weight, axis=axis, keepdims=keepdims)
            sumwxy = sum((x - xmean)*(y - ymean)*weight,
                         axis=axis,
                         keepdims=keepdims)
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
        intercept       = numpy.true_divide(((sumwxx*sumwy) - (sumwx*sumwxy)),
                                            delta)
        slope           = numpy.true_divide(((sumw*sumwxy) - (sumwx*sumwy)),
                                            delta)
        intercept_error = numpy.sqrt(numpy.true_divide(sumwxx, delta))
        slope_error     = numpy.sqrt(numpy.true_divide(sumw, delta))

        intercept       = awkward1.operations.convert.tolayout(
                            intercept, allowrecord=True, allowother=True)
        slope           = awkward1.operations.convert.tolayout(
                            slope, allowrecord=True, allowother=True)
        intercept_error = awkward1.operations.convert.tolayout(
                            intercept_error, allowrecord=True, allowother=True)
        slope_error     = awkward1.operations.convert.tolayout(
                            slope_error, allowrecord=True, allowother=True)

        scalar = (not isinstance(intercept, awkward1.layout.Content) and
                  not isinstance(slope, awkward1.layout.Content) and
                  not isinstance(intercept_error, awkward1.layout.Content) and
                  not isinstance(slope_error, awkward1.layout.Content))

        if not isinstance(intercept, (awkward1.layout.Content,
                                      awkward1.layout.Record)):
            intercept = awkward1.layout.NumpyArray(numpy.array([intercept]))
        if not isinstance(slope, (awkward1.layout.Content,
                                  awkward1.layout.Record)):
            slope = awkward1.layout.NumpyArray(numpy.array([slope]))
        if not isinstance(intercept_error, (awkward1.layout.Content,
                                            awkward1.layout.Record)):
            intercept_error = awkward1.layout.NumpyArray(
                                numpy.array([intercept_error]))
        if not isinstance(slope_error, (awkward1.layout.Content,
                                        awkward1.layout.Record)):
            slope_error = awkward1.layout.NumpyArray(
                            numpy.array([slope_error]))

        out = awkward1.layout.RecordArray([intercept,
                                           slope,
                                           intercept_error,
                                           slope_error],
                                          ["intercept",
                                           "slope",
                                           "intercept_error",
                                           "slope_error"])
        out.setparameter("__record__", "LinearFit")
        if scalar:
            out = out[0]

        return awkward1._util.wrap(out, awkward1._util.behaviorof(x, y))

def softmax(x, axis=None, keepdims=False):
    with numpy.errstate(invalid="ignore"):
        expx = numpy.exp(x)
        denom = sum(expx, axis=axis, keepdims=keepdims)
        return numpy.true_divide(expx, denom)

__all__ = [x for x in list(globals())
             if not x.startswith("_") and
             x not in ("collections", "numpy", "awkward1")]
