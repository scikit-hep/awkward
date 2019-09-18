import ctypes
import math
import itertools
import functools
import operator
import collections.abc

import numpy

class NumpyArray:
    def __init__(self, array):
        assert len(array.shape) == len(array.strides)

        minpos, nbytes = 0, 0
        for i in range(len(array.shape)):
            if array.strides[i] < 0:
                minpos += (array.shape[i] - 1)*array.strides[i]
                nbytes -= array.shape[i]*array.strides[i]
            else:
                nbytes += array.shape[i]*array.strides[i]

        self.ptr = numpy.ctypeslib.as_array(ctypes.cast(array.ctypes.data + minpos, ctypes.POINTER(ctypes.c_uint8)), (nbytes,))
        self.shape = array.shape
        self.strides = array.strides
        self.itemsize = array.itemsize
        self.dtype = array.dtype
        self.byteoffset = -minpos

    def copy(self, ptr=None, shape=None, strides=None, itemsize=None, dtype=None, byteoffset=None):
        out = type(self).__new__(type(self))
        out.ptr = self.ptr
        out.shape = self.shape
        out.strides = self.strides
        out.itemsize = self.itemsize
        out.dtype = self.dtype
        out.byteoffset = self.byteoffset
        if ptr is not None:
            out.ptr = ptr
        if shape is not None:
            out.shape = shape
        if strides is not None:
            out.strides = strides
        if itemsize is not None:
            out.itemsize = itemsize
        if dtype is not None:
            out.dtype = dtype
        if byteoffset is not None:
            out.byteoffset = byteoffset
        return out

    def __array__(self):
        assert len(self.shape) == len(self.strides)
        if len(self.shape) == 0:
            return numpy.frombuffer(self.ptr[self.byteoffset : self.byteoffset + self.itemsize], dtype=self.dtype).reshape(())
        else:
            return numpy.lib.stride_tricks.as_strided(self.ptr[self.byteoffset : self.byteoffset + self.itemsize].view(self.dtype), self.shape, self.strides)

    def tolist(self):
        return numpy.array(self).tolist()

    def __len__(self):
        return self.shape[0]

    @property
    def isscalar(self):
        return len(self.shape) == 0

    @property
    def iscontiguous(self):
        test = self.itemsize
        for sh, st in zip(self.shape[::-1], self.strides[::-1]):
            if st != test:
                return False
            test *= sh
        return True   # isscalar implies iscontiguous

    def become_contiguous(self):
        out = self.contiguous()
        self.ptr = out.ptr
        self.shape = out.shape
        self.strides = out.strides
        self.itemsize = out.itemsize
        self.dtype = out.dtype
        self.byteoffset = out.byteoffset

    def contiguous(self):
        if self.iscontiguous:
            return self
        else:
            bytepos = numpy.arange(0, self.shape[0]*self.strides[0], self.strides[0])
            return self.contiguous_next(bytepos)

    def contiguous_next(self, bytepos):
        if self.iscontiguous:
            ptr = numpy.full(len(bytepos)*self.strides[0], 123, dtype=numpy.uint8)
            for i in range(len(bytepos)):
                ptr[i*self.strides[0] : (i + 1)*self.strides[0]] = self.ptr[self.byteoffset + bytepos[i] : self.byteoffset + bytepos[i] + self.strides[0]]
            return self.copy(ptr=ptr, byteoffset=0)

        elif len(self.shape) == 1:
            ptr = numpy.full(len(bytepos)*self.itemsize, 123, dtype=numpy.uint8)
            for i in range(len(bytepos)):
                ptr[i*self.itemsize : (i + 1)*self.itemsize] = self.ptr[self.byteoffset + bytepos[i] : self.byteoffset + bytepos[i] + self.itemsize]
            return self.copy(ptr=ptr, strides=(self.itemsize,), byteoffset=0)

        else:
            next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))
            nextbytepos = numpy.full(len(bytepos)*self.shape[1], 999, dtype=int)
            for i in range(len(bytepos)):
                for j in range(self.shape[1]):
                    nextbytepos[i*self.shape[1] + j] = bytepos[i] + j*self.strides[1]
            out = next.contiguous_next(nextbytepos)
            return out.copy(shape=self.shape, strides=(self.shape[1]*out.strides[0],) + out.strides)

    def __getitem__(self, where):
        assert len(self.shape) != 0

        if not isinstance(where, tuple):
            where = (where,)

        if False and all(x is numpy.newaxis or x is Ellipsis or (isinstance(x, tuple) and len(x) == 0) or isinstance(x, (int, numpy.integer, slice)) for x in where):
            nexthead, nexttail = head_tail(where)
            return getitem_bystrides(nexthead, nexttail)

        else:
            self.become_contiguous()

            broadcastable, broadcastable_i = [], []
            for i, x in enumerate(where):
                if not isinstance(x, tuple) and isinstance(x, (int, numpy.integer, collections.abc.Iterable)):
                    broadcastable.append(x)
                    broadcastable_i.append(i)
            broadcasted = broadcast_arrays(broadcastable)

            where = tuple(broadcasted[broadcastable_i[i]] if i in broadcastable_i else x for i, x in enumerate(where))
            print(where)

            raise Exception
            
            where = tuple(numpy.array(x) if isinstance(x, collections.abc.Iterable) and not (isinstance(x, tuple) and len(x) == 0) else x for x in where)

            advlen = -1
            for x in where:
                if isinstance(x, numpy.ndarray) and not (len(x) == 1 and issubclass(x.dtype.type, numpy.integer)):
                    advlen = len(x)
                    break

            originalhead, originaltail = head_tail(where)

            if advlen != -1:
                where = tuple(numpy.repeat(x, advlen) if isinstance(x, (int, numpy.integer)) or (isinstance(x, numpy.ndarray) and len(x) == 1 and issubclass(x.dtype.type, numpy.integer)) else x for x in where)

            nexthead, nexttail = head_tail(where)

            nextcarry = numpy.array([0])
            nextadvanced = None
            length = 1
            next = self.copy(shape=(1,) + self.shape, strides=(self.shape[0]*self.strides[0],) + self.strides)
            out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length, next.strides[0])
            return out.copy(shape=out.shape[1:], strides=out.strides[1:])

    def getitem_next(self, head, tail, carry, advanced, length, stride):
        assert len(self.shape) == len(self.strides)

        if head is numpy.newaxis:
            raise NotImplementedError("numpy.newaxis")

        elif head is Ellipsis:
            raise NotImplementedError("...")

        elif isinstance(head, tuple) and len(head) == 0:
            ptr = numpy.full(len(carry)*stride, 123, dtype=numpy.uint8)
            for i in range(len(carry)):
                ptr[i*stride : (i + 1)*stride] = self.ptr[self.byteoffset + carry[i]*stride : self.byteoffset + (carry[i] + 1)*stride]
            shape = (len(carry),) + self.shape[1:]
            strides = (stride,) + self.strides[1:]
            return self.copy(ptr=ptr, shape=shape, strides=strides, byteoffset=0)

        elif isinstance(head, (int, numpy.integer)):
            assert len(self.shape) >= 2
            next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))

            nexthead, nexttail = head_tail(tail)
            nextcarry = numpy.full(len(carry), 999, dtype=int)

            skip, remainder = divmod(self.strides[0], self.strides[1])
            assert remainder == 0
            for i in range(len(carry)):
                nextcarry[i] = skip*carry[i] + head

            out = next.getitem_next(nexthead, nexttail, nextcarry, advanced, length, next.strides[0])
            shape = (length,) + out.shape[1:]
            return out.copy(shape=shape)

        elif isinstance(head, slice):
            assert len(self.shape) >= 2
            next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))

            start, stop, step = head.start, head.stop, head.step
            if step is None:
                step = 1
            assert step != 0
            d, m = divmod(abs(start - stop), abs(step))
            headlen = d + (1 if m != 0 else 0)

            nexthead, nexttail = head_tail(tail)
            nextcarry = numpy.full(len(carry)*headlen, 999, dtype=int)

            skip, remainder = divmod(self.strides[0], self.strides[1])
            assert remainder == 0

            if advanced is None:
                nextadvanced = None
                for i in range(len(carry)):
                    for j in range(headlen):
                        nextcarry[i*headlen + j] = skip*carry[i] + head.start + j*step

                out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*headlen, next.strides[0])
                shape = (length, headlen) + out.shape[1:]
                strides = (shape[1]*out.strides[0],) + out.strides
                return out.copy(shape=shape, strides=strides)

            else:
                nextadvanced = numpy.full(len(carry)*headlen, 999, dtype=int)
                for i in range(len(carry)):
                    for j in range(headlen):
                        nextcarry[i*headlen + j] = skip*carry[i] + head.start + j*step
                        nextadvanced[i*headlen + j] = advanced[i]

                out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*headlen, next.strides[0])
                shape = (length, headlen) + out.shape[1:]
                strides = (shape[1]*out.strides[0],) + out.strides
                return out.copy(shape=shape, strides=strides)

        else:
            head = numpy.asarray(head)
            if issubclass(head.dtype.type, numpy.integer):
                assert len(self.shape) >= 2
                next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))

                nexthead, nexttail = head_tail(tail)

                skip, remainder = divmod(self.strides[0], self.strides[1])
                assert remainder == 0

                if advanced is None:
                    nextcarry = numpy.full(len(carry)*len(head), 999, dtype=int)
                    nextadvanced = numpy.full(len(carry)*len(head), 999, dtype=int)
                    for i in range(len(carry)):
                        for j in range(len(head)):
                            nextcarry[i*len(head) + j] = skip*carry[i] + head[j]
                            nextadvanced[i*len(head) + j] = j

                    out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*len(head), next.strides[0])
                    shape = (length, len(head)) + out.shape[1:]
                    strides = (shape[1]*out.strides[0],) + out.strides
                    return out.copy(shape=shape, strides=strides)

                else:
                    nextcarry = numpy.full(len(carry), 999, dtype=int)
                    nextadvanced = numpy.full(len(carry), 999, dtype=int)
                    for i in range(len(carry)):
                        nextcarry[i] = skip*carry[i] + head[advanced[i]]
                        nextadvanced[i] = advanced[i]

                    out = next.getitem_next(nexthead, nexttail, nextcarry, nextadvanced, length*len(head), next.strides[0])
                    shape = (length,) + out.shape[1:]
                    return out.copy(shape=shape)

            elif issubclass(head.dtype.type, (numpy.bool_, numpy.bool)):
                raise NotImplementedError("boolarray")

            else:
                raise TypeError("cannot use {0} as an index".format(head))

def head_tail(x):
    head = () if len(x) == 0 else x[0]
    tail = x[1:]
    return head, tail

def flatten_shape(shape):
    if len(shape) == 1:
        return ()
    else:
        return (shape[0]*shape[1],) + shape[2:]

def flatten_strides(strides):
    return strides[1:]

def broadcast_arrays(*args):
    return numpy.broadcast_arrays(*args)

# a = numpy.arange(2*1*3*1).reshape(2, 1, 3, 1)
# print(a.tolist())
# b = NumpyArray(a)
# cut = (slice(0, 2), numpy.array([0, 0, 0, 0]), slice(0, 3), numpy.array([0, 0, 0, 0]))
# acut = a[cut]
# bcut = b[cut]
# print("should be shape", acut.shape, "strides", acut.strides)
# print("       is shape", bcut.shape, "strides", bcut.strides)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

# a = numpy.arange(7*5).reshape(7, 5)
# print(a.tolist())
# b = NumpyArray(a)
# cut = (numpy.array([2, 0, 0, 1]), numpy.array([0, 1, 2, 0]),)
# acut = a[cut]
# bcut = b[cut]
# print("should be shape", acut.shape, "strides", acut.strides)
# print("       is shape", bcut.shape, "strides", bcut.strides)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

# a = numpy.arange(7*5*6*8).reshape(7, 5, 6, 8)
# b = NumpyArray(a)
# cut = (slice(0, 5), 0, slice(1, 4), numpy.array([1, 0, 0, 1]))
# acut = a[cut]
# bcut = b[cut]
# print("should be shape", acut.shape, "strides", acut.strides)
# print("       is shape", bcut.shape, "strides", bcut.strides)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

# # a = numpy.arange(7*5).reshape(7, 5)
# # a = numpy.arange(7*5*6).reshape(7, 5, 6)
# a = numpy.arange(7*5*6*8).reshape(7, 5, 6, 8)
# b = NumpyArray(a)
# # for depth in 1, 2:
# #     for cuts in itertools.permutations((0, 1, slice(0, 5), slice(1, 4), slice(2, 3)), depth):
# # for depth in 1, 2, 3:
# #     for cuts in itertools.permutations((0, 1, 2, slice(0, 5), slice(1, 4), slice(2, 3)), depth):
# for depth in 1, 2, 3, 4:
#     for cuts in itertools.permutations((0, 1, 2, 3, slice(0, 5), slice(1, 4), slice(1, 4), slice(1, 4), slice(2, 0, -1), slice(2, 0, -1), numpy.array([1, 0, 0, 1]), numpy.array([2, 2, 0, 1])), depth):
#         doit = True
#         for i in range(max(0, len(cuts) - 2)):
#             if isinstance(cuts[i], (int, numpy.ndarray)) and isinstance(cuts[i + 1], slice) and isinstance(cuts[i + 2], (int, numpy.ndarray)):
#                 doit = False
#         if doit:
#             print(cuts)
#             acut = a[cuts].tolist()
#             bcut = b[cuts].tolist()
#             # print(acut)
#             # print(bcut)
#             # print()
#             assert acut == bcut

# a = numpy.arange(7*5*6*8).reshape(7, 5, 6, 8)[::2, ::3, ::-1, ::-2]
# b = NumpyArray(a)
# assert a.tolist() == b.tolist()
# b.become_contiguous()
# assert a.tolist() == b.tolist()
