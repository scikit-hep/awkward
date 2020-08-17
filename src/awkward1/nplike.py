# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy


def of(*arrays):
    return Numpy.instance()


class NumpyLike(object):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Numpy()
        return cls._instance

class NumpyMetadata(NumpyLike):
    bool = numpy.bool
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128

    intp = numpy.intp
    integer = numpy.integer
    floating = numpy.floating
    number = numpy.number
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    errstate = numpy.errstate
    newaxis = numpy.newaxis

if hasattr(numpy, "float16"):
    NumpyLike.float16 = numpy.float16

if hasattr(numpy, "float128"):
    NumpyLike.float128 = numpy.float128

if hasattr(numpy, "complex256"):
    NumpyLike.complex256 = numpy.complex256

if hasattr(numpy, "datetime64"):
    NumpyLike.datetime64 = numpy.datetime64

if hasattr(numpy, "timedelta64"):
    NumpyLike.timedelta64 = numpy.timedelta64


class Numpy(NumpyLike):
    def __init__(self):
        self._module = numpy

    def __getattr__(self, name):
        return getattr(self._module, name)


######################### numpy.all
#         return reduce([numpy.all(x) for x in awkward1._util.completely_flatten(layout)])

######################### numpy.any
#         return reduce([numpy.any(x) for x in awkward1._util.completely_flatten(layout)])

######################### numpy.arange
#             [numpy.arange(len(self.arrayptrs)), self.arrayptrs, self.sharedptrs]
#                         numpy.arange(start, start + len(x), dtype=numpy.int64)
#     parents = numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
#             index = numpy.arange(len(akcondition), dtype=numpy.int64)
#                 *[numpy.arange(len(x), dtype=numpy.int64) for x in layouts],
#             offsets = numpy.arange(0, (len(layout) + 1) * layout.size, layout.size)
#             index = numpy.arange(len(tags), dtype=numpy.int32)
#                     numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
#                 numpy.arange(offsets[-1], dtype=counts.dtype)
#                 index[mask] = numpy.arange(numpy.count_nonzero(mask))
#             index[~mask] = numpy.arange(
#                 nextindex = numpy.arange(len(mask), dtype=numpy.int64)
#                                     numpy.arange(len(x), dtype=numpy.int64), maxsize

######################### numpy.argmax
#                     out = numpy.argmax(tmp, axis=None)
#                 out = numpy.argmax(tmp, axis=None)

######################### numpy.argmin
#                     out = numpy.argmin(tmp, axis=None)
#                 out = numpy.argmin(tmp, axis=None)

######################### numpy.array
#         buf = dynamic_addrs[pyvalue] = numpy.array(pyvalue.encode("utf-8") + b"\x00")
#         self.arrayptrs = numpy.array([arrayptr(x) for x in positions], dtype=numpy.intp)
#         self.sharedptrs = numpy.array(
#         return numpy.array(out, *args, **kwargs)
#         return numpy.array(awkward1.operations.structure.is_none(self))
#             y = awkward1.layout.NumpyArray(numpy.array([y]))
#                 return apply(awkward1.layout.NumpyArray(numpy.array([])))
#                 index = numpy.array(numpy.asarray(layout.index), copy=True)
#                     numpy.array([0, 0], dtype=numpy.int64)
#             return apply(awkward1.layout.NumpyArray(numpy.array([])))
#             intercept = awkward1.layout.NumpyArray(numpy.array([intercept]))
#             slope = awkward1.layout.NumpyArray(numpy.array([slope]))
#             intercept_error = awkward1.layout.NumpyArray(numpy.array([intercept_error]))
#             slope_error = awkward1.layout.NumpyArray(numpy.array([slope_error]))
#         return numpy.array([array])[0]
#         return numpy.array([array])[0]
#         return numpy.array(
#         return numpy.array(
#         return numpy.array([])
#                     values.append(awkward1.layout.NumpyArray(numpy.array([x])))
#                     values.append(awkward1.layout.NumpyArray(numpy.array([x])))
#             return numpy.array([])
#             index = numpy.array(layout.index, copy=True)
#             applicable_indices = numpy.array(index)[numpy.equal(tags, i)]
#                 numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)
#         return (numpy.array([], dtype=numpy.bool_),)
#                     else awkward1.layout.NumpyArray(numpy.array([], dtype=numpy.bool_))
#         return awkward1._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)
#         return awkward1._connect._numpy.array_function(func, types, args, kwargs)
#         return awkward1._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)
#         return awkward1._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)
#         return awkward1._connect._numpy.array_function(func, types, args, kwargs)

######################### numpy.array_equal
#                 numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)

######################### numpy.asarray
#     t = numba.typeof(numpy.asarray(obj))
#         numba.typeof(numpy.asarray(obj.starts)),
#         numba.typeof(numpy.asarray(obj.index)),
#         numba.typeof(numpy.asarray(obj.index)),
#         numba.typeof(numpy.asarray(obj.mask)),
#         numba.typeof(numpy.asarray(obj.mask)),
#         numba.typeof(numpy.asarray(obj.tags)),
#         numba.typeof(numpy.asarray(obj.index)),
#             arrays.append(numpy.asarray(layout.identities))
#             lookup.arrayptr[pos + self.IDENTITIES] = numpy.asarray(
#         array = numpy.asarray(layout)
#         lookup.original_positions[pos + self.ARRAY] = numpy.asarray(layout)
#             starts = numpy.asarray(layout.starts)
#             stops = numpy.asarray(layout.stops)
#             offsets = numpy.asarray(layout.offsets)
#             starts = numpy.asarray(layout.starts)
#             stops = numpy.asarray(layout.stops)
#             offsets = numpy.asarray(layout.offsets)
#         arrays.append(numpy.asarray(layout.index))
#         index = numpy.asarray(layout.index)
#         arrays.append(numpy.asarray(layout.index))
#         index = numpy.asarray(layout.index)
#         arrays.append(numpy.asarray(layout.mask))
#         mask = numpy.asarray(layout.mask)
#         arrays.append(numpy.asarray(layout.mask))
#         mask = numpy.asarray(layout.mask)
#         arrays.append(numpy.asarray(layout.tags))
#         arrays.append(numpy.asarray(layout.index))
#         tags = numpy.asarray(layout.tags)
#         index = numpy.asarray(layout.index)
#                 numpy.asarray(layout.stops, dtype=numpy.intp),
#                                 numpy.asarray(x._value), x._trace, x._node
#             lambda x: autograd.numpy.numpy_vspaces.ArrayVSpace(numpy.asarray(x)),
#             indices = numpy.asarray(indices, dtype=numpy.int64)
#         tmp = numpy.asarray(self.layout)
#         tmp = numpy.asarray(self.layout)
#     counts1 = numpy.asarray(one.count(axis=-1))
#     counts2 = numpy.asarray(two.count(axis=-1))
#     offsets = numpy.asarray(offsets)
#             m = numpy.asarray(layoutmask)
#             tags = numpy.asarray(akcondition) == 0
#                 tags = numpy.asarray(layout.tags)
#                 index = numpy.array(numpy.asarray(layout.index), copy=True)
#                         bigmask[tags == tag] = numpy.asarray(content.bytemask()).view(
#             tags = numpy.asarray(layout.tags)
#             return numpy.asarray(layout.bytemask()).view(numpy.bool_)
#             nulls = numpy.asarray(layout.bytemask()).view(numpy.bool_)
#                 sizes.extend(numpy.asarray(layout).shape[1:])
#                 sizes.extend(numpy.asarray(layout).shape[1 : axis + 2])
#         tags = numpy.asarray(array.tags)
#         mask0 = numpy.asarray(array.bytemask()).view(numpy.bool_)
#         return numpy.asarray(array)
#         return numpy.asarray(array)
#         return numpy.asarray(array).tolist()
#             return numpy.asarray(layout)
#                 numpy.asarray(layout.starts),
#                 numpy.asarray(layout.stops),
#                 numpy.asarray(layout.starts),
#                 numpy.asarray(layout.stops),
#                 numpy.asarray(layout.starts),
#                 numpy.asarray(layout.stops),
#                 numpy.asarray(layout.offsets), recurse(layout.content)
#                 numpy.asarray(layout.offsets), recurse(layout.content)
#                 numpy.asarray(layout.offsets), recurse(layout.content)
#                 numpy.asarray(layout.tags),
#                 numpy.asarray(layout.index),
#                 numpy.asarray(layout.tags),
#                 numpy.asarray(layout.index),
#                 numpy.asarray(layout.tags),
#                 numpy.asarray(layout.index),
#             index = numpy.asarray(layout.index)
#             index = numpy.asarray(layout.index)
#                 numpy.asarray(layout.index), recurse(layout.content)
#                 numpy.asarray(layout.index), recurse(layout.content)
#                 numpy.asarray(layout.index), recurse(layout.content)
#                 numpy.asarray(layout.mask),
#                 numpy.asarray(layout.mask),
#             numpy_arr = numpy.asarray(layout)
#             offsets = numpy.asarray(layout.offsets, dtype=numpy.int32)
#             offsets = numpy.asarray(layout.offsets)
#             offsets = numpy.asarray(layout.offsets, dtype=numpy.int64)
#                         pyarrow.py_buffer(numpy.asarray(layout.tags)),
#                             numpy.asarray(layout.index).astype(numpy.int32)
#                         pyarrow.py_buffer(numpy.asarray(layout.tags)),
#                             numpy.asarray(layout.index).astype(numpy.int32)
#             index = numpy.asarray(layout.index)
#             bitmask = numpy.asarray(layout.mask, dtype=numpy.uint8)
#             mask = numpy.asarray(layout.mask, dtype=numpy.bool) == layout.valid_when
#             array = numpy.asarray(layout)
#                 numpy.asarray(layout.index)
#                 numpy.asarray(layout.index)
#                 numpy.asarray(layout.mask)
#                 numpy.asarray(layout.mask)
#                 numpy.asarray(layout.starts)
#                 numpy.asarray(layout.stops)
#                 numpy.asarray(layout.offsets)
#             array = numpy.asarray(layout)
#                 numpy.asarray(layout.tags)
#                 numpy.asarray(layout.index)
#             offsets = numpy.asarray(offsets)
#         return (numpy.asarray(array),)
#                     tagslist.append(numpy.asarray(x.tags))
#                     m = numpy.asarray(x.bytemask()).view(numpy.bool_)
#                 numpy.asarray(layout), layout.identities, None

######################### numpy.atleast_1d
#     return numpy.atleast_1d(*[awkward1.operations.convert.to_numpy(x) for x in arrays])

######################### numpy.bitwise_or
#                         numpy.bitwise_or(mask, m, out=mask)

######################### numpy.broadcast_to
#                 mask = numpy.broadcast_to(

######################### numpy.ceil
#                         int(numpy.ceil(len(numpy_arr) / 8.0)) * 8, dtype=numpy_arr.dtype
#                 length = int(numpy.ceil(len(nulls) / 8.0)) * 8

######################### numpy.concatenate
#                 return numpy.concatenate(tocat)
#             out = awkward1.layout.NumpyArray(numpy.concatenate(out))
#             return numpy.concatenate(tocat)
#                 out = numpy.concatenate(contents)

######################### numpy.count_nonzero
#             [numpy.count_nonzero(x) for x in awkward1._util.completely_flatten(layout)]
#                 index[mask] = numpy.arange(numpy.count_nonzero(mask))
#                 len(mask) - numpy.count_nonzero(mask), dtype=numpy.int64

######################### numpy.cumsum
#             numpy.cumsum(offsets, out=offsets)

######################### numpy.empty
#             index = numpy.empty(len(tags), dtype=numpy.int64)
#                 bigmask = numpy.empty(len(index), dtype=numpy.bool_)
#             out = numpy.empty(len(layout), dtype=numpy.bool_)
#         data = numpy.empty(shape, dtype=content.dtype)
#             return numpy.empty(len(array), dtype=[])
#         out = numpy.empty(
#                     ready_to_pack = numpy.empty(
#                 this_bytemask = numpy.empty(length, dtype=numpy.uint8)
#             tags = numpy.empty(length, dtype=numpy.int8)
#             index = numpy.empty(length, dtype=numpy.int64)
#             out = numpy.empty(len(self._layout), dtype="O")
#                 out = numpy.empty(len(self._layout), dtype="O")

######################### numpy.equal
# awkward1.behavior[numpy.equal, "bytestring", "bytestring"] = _string_equal
# awkward1.behavior[numpy.equal, "string", "string"] = _string_equal
#             applicable_indices = numpy.array(index)[numpy.equal(tags, i)]

######################### numpy.exp
#         expx = numpy.exp(x)

######################### numpy.frombuffer
#         voidptr = numpy.frombuffer(pyptr, dtype=numpy.intp).item()
#         voidptr = numpy.frombuffer(pyptr, dtype=numpy.intp).item()
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#                 numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)[: length + 1]
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#                 numpy.frombuffer(buffers.pop(0), dtype=numpy.int64)[: length + 1]
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             tags = numpy.frombuffer(buffers.pop(0), dtype=numpy.int8)[:length]
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             tags = numpy.frombuffer(buffers.pop(0), dtype=numpy.int8)[:length]
#             index = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)[:length]
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)
#             contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int64)
#             contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)
#             contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int64)
#             contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#             out = numpy.frombuffer(data, dtype=numpy.uint8)
#                     numpy.frombuffer(mask, dtype=numpy.uint8)
#                 mask = numpy.frombuffer(mask, dtype=numpy.uint8)
#                 numpy.frombuffer(buffers.pop(0), dtype=tpe.to_pandas_dtype())[:length]
#                     numpy.frombuffer(mask, dtype=numpy.uint8)

######################### numpy.full
#             index = numpy.full(len(mask), -1, dtype=numpy.int64)
#             index = numpy.full(maxlen, x.at, dtype=numpy.int64)

######################### numpy.logical_and
#     possible = numpy.logical_and(out, counts1)

######################### numpy.ma
#             if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
#                 return numpy.ma.concatenate(tocat)
#         if any(isinstance(x, numpy.ma.MaskedArray) for x in out):
#             out = awkward1.layout.NumpyArray(numpy.ma.concatenate(out))
#         return reduce([numpy.max(x) for x in tmp if len(x) > 0])
#     if isinstance(array, numpy.ma.MaskedArray):
#         mask = numpy.ma.getmask(array)
#         array = numpy.ma.getdata(array)
#         if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
#             return numpy.ma.concatenate(tocat)
#         if any(isinstance(x, numpy.ma.MaskedArray) for x in contents):
#                 out = numpy.ma.concatenate(contents)
#                     "cannot convert {0} into numpy.ma.MaskedArray".format(array)
#             return numpy.ma.MaskedArray(content)
#                 if isinstance(content, numpy.ma.MaskedArray):
#                     mask1 = numpy.ma.getmaskarray(content)
#                 return numpy.ma.MaskedArray(data, mask)
#                 return numpy.ma.MaskedArray(content)
#                         numpy.ma.MaskedArray,
#                         numpy.ma.MaskedArray,
#         elif isinstance(array, numpy.ma.MaskedArray):
#     elif isinstance(array, numpy.ma.MaskedArray):
#         mask = numpy.ma.getmask(array)
#         data = numpy.ma.getdata(array)
#             if len(offsets) == 0 or numpy.max(offsets) <= numpy.iinfo(numpy.int32).max:
#             numpy.max(index) + 1,
#             numpy.max(index) + 1,
#                 numpy.max(applicable_indices) + 1,

######################### numpy.max
#         return reduce([numpy.max(x) for x in tmp if len(x) > 0])
#             if len(offsets) == 0 or numpy.max(offsets) <= numpy.iinfo(numpy.int32).max:
#             numpy.max(index) + 1,
#             numpy.max(index) + 1,
#                 numpy.max(applicable_indices) + 1,

######################### numpy.meshgrid
#             for x in numpy.meshgrid(

######################### numpy.min
#         return reduce([numpy.min(x) for x in tmp if len(x) > 0])

######################### numpy.ndarray
#     elif isinstance(array, numpy.ndarray):
#         assert isinstance(out, tuple) and all(isinstance(x, numpy.ndarray) for x in out)
#         if isinstance(mask, numpy.ndarray) and len(mask.shape) > 1:
#     elif isinstance(array, numpy.ndarray):
#                 raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
#             raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
#         raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
#     elif isinstance(array, numpy.ndarray):
#     elif isinstance(array, numpy.ndarray):
#                         numpy.ndarray,
#                         numpy.ndarray,
#         elif isinstance(array, numpy.ndarray):
#     elif isinstance(array, numpy.ndarray):
#         elif isinstance(data, numpy.ndarray) and data.dtype != numpy.dtype("O"):

######################### numpy.nonzero
#         out = numpy.nonzero(awkward1.operations.convert.to_numpy(akcondition))

######################### numpy.ones
#             offsets = numpy.ones(len(layout) + 1, dtype=numpy.int64)

######################### numpy.packbits
#                 numpy_arr = numpy.packbits(
#             this_bitmask = numpy.packbits(
#                 bitmask = numpy.packbits(
#             bitmask = numpy.packbits(bytemask.reshape(-1, 8)[:, ::-1].reshape(-1))

######################### numpy.prod
#             [numpy.prod(x) for x in awkward1._util.completely_flatten(layout)]

######################### numpy.repeat
#     parents = numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
#                     what = awkward1.layout.NumpyArray(numpy.repeat(what, len(base)))
#                     numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
#                 newrows = [numpy.repeat(x, counts) for x in row_arrays]
#                 - numpy.repeat(starts, counts)
#                                 numpy.repeat(

######################### numpy.searchsorted
#             return numpy.searchsorted(stops, where, side="right")

######################### numpy.size
#             [numpy.size(x) for x in awkward1._util.completely_flatten(layout)]

######################### numpy.sqrt
#         return numpy.sqrt(
#         return numpy.true_divide(sumwxy, numpy.sqrt(sumwxx * sumwyy))
#         intercept_error = numpy.sqrt(numpy.true_divide(sumwxx, delta))
#         slope_error = numpy.sqrt(numpy.true_divide(sumw, delta))

######################### numpy.stack
#             combos = numpy.stack(tagslist, axis=-1)

######################### numpy.sum
#         return reduce([numpy.sum(x) for x in awkward1._util.completely_flatten(layout)])

######################### numpy.true_divide
#         return numpy.true_divide(sumwxn, sumw)
#         return numpy.true_divide(sumwx, sumw)
#             return numpy.true_divide(sumwxx, sumw) * numpy.true_divide(
#             return numpy.true_divide(sumwxx, sumw)
#         return numpy.true_divide(sumwxy, sumw)
#         return numpy.true_divide(sumwxy, numpy.sqrt(sumwxx * sumwyy))
#         intercept = numpy.true_divide(((sumwxx * sumwy) - (sumwx * sumwxy)), delta)
#         slope = numpy.true_divide(((sumw * sumwxy) - (sumwx * sumwy)), delta)
#         intercept_error = numpy.sqrt(numpy.true_divide(sumwxx, delta))
#         slope_error = numpy.sqrt(numpy.true_divide(sumw, delta))
#         return numpy.true_divide(expx, denom)

######################### numpy.unique
#             for tag, combo in enumerate(numpy.unique(combos)):

######################### numpy.unpackbits
#                     numpy.unpackbits(~mask)
#                     numpy.unpackbits(bitmask).reshape(-1, 8)[:, ::-1].reshape(-1)
#             out = numpy.unpackbits(out).reshape(-1, 8)[:, ::-1].reshape(-1)

######################### numpy.vstack
#         return numpy.vstack(

######################### numpy.zeros
#             return numpy.zeros(len(layout), dtype=numpy.bool_)
#             bytemask = numpy.zeros(
