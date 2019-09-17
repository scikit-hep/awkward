import ctypes
import math
import itertools
import functools
import operator

import numpy

class NumpyArray:
    def __init__(self, array):
        assert len(array.shape) == len(array.strides)

        minpos, maxpos = 0, 0
        for i in range(len(array.shape)):
            if array.strides[i] < 0:
                minpos += (array.shape[i] - 1)*array.strides[i]
            else:
                maxpos += array.shape[i]*array.strides[i]

        self.ptr = numpy.ctypeslib.as_array(ctypes.cast(array.ctypes.data + minpos, ctypes.POINTER(ctypes.c_uint8)), (maxpos - minpos,))
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
            # ptr = self.ptr[self.byteoffset : self.byteoffset + self.strides[0]*self.shape[0]]
            # return numpy.lib.stride_tricks.as_strided(numpy.frombuffer(ptr, dtype=self.dtype), self.shape, self.strides)
            return numpy.lib.stride_tricks.as_strided(self.ptr[self.byteoffset : self.byteoffset + self.itemsize].view(self.dtype), self.shape, self.strides)

    def tolist(self):
        return numpy.array(self).tolist()

    def __len__(self):
        return self.shape[0]

    @property
    def isscalar(self):
        return len(self.shape) == 0

    @property
    def iscompact(self):
        # isscalar implies iscompact
        test = self.itemsize
        for sh, st in zip(self.shape[::-1], self.strides[::-1]):
            if st != test:
                return False
            test *= sh
        return True

    def compact(self):
        if self.iscompact:
            return self
        else:
            bytepos = numpy.arange(0, self.shape[0]*self.strides[0], self.strides[0])
            return self.compact_next(bytepos)

    def compact_next(self, bytepos):
        if self.iscompact:
            ptr = numpy.full(len(bytepos)*self.strides[0], 123, dtype=numpy.uint8)
            for i in range(len(bytepos)):
                print("to", i*self.strides[0], ":", (i + 1)*self.strides[0], "from", self.byteoffset + bytepos[i], ":", self.byteoffset + bytepos[i] + self.strides[0], "which is", self.ptr[self.byteoffset + bytepos[i] : self.byteoffset + bytepos[i] + self.strides[0]])
                ptr[i*self.strides[0] : (i + 1)*self.strides[0]] = self.ptr[self.byteoffset + bytepos[i] : self.byteoffset + bytepos[i] + self.strides[0]]
            return self.copy(ptr=ptr, byteoffset=0)

        elif len(self.shape) == 1:
            ptr = numpy.full(len(bytepos)*self.itemsize, 123, dtype=numpy.uint8)
            for i in range(len(bytepos)):
                print("to", i*self.itemsize, ":", (i + 1)*self.itemsize, "from", self.byteoffset + bytepos[i], ":", self.byteoffset + bytepos[i] + self.itemsize, "which is", self.ptr[self.byteoffset + bytepos[i] : self.byteoffset + bytepos[i] + self.itemsize])

                ptr[i*self.itemsize : (i + 1)*self.itemsize] = self.ptr[self.byteoffset + bytepos[i] : self.byteoffset + bytepos[i] + self.itemsize]
            return self.copy(ptr=ptr, strides=(self.itemsize,), byteoffset=0)

        else:
            next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))
            nextbytepos = numpy.full(len(bytepos)*self.shape[1], 999, dtype=int)
            for i in range(len(bytepos)):
                for j in range(self.shape[1]):
                    nextbytepos[i*self.shape[1] + j] = bytepos[i] + j*self.strides[1]
            out = next.compact_next(nextbytepos)
            return out.copy(shape=self.shape, strides=(self.shape[1]*out.strides[0],) + out.strides)

    def __getitem__(self, where):
        assert len(self.shape) != 0

        if not isinstance(where, tuple):
            where = (where,)

        # if any strides are not an even multiple of their underlying stride
        # or any strides are negative, compact before doing a getitem

        next = self.copy(shape=(1,) + self.shape, strides=(self.shape[0]*self.strides[0],) + self.strides)
        nexthead, nexttail = head_tail(where)
        nextcarry = numpy.array([0])
        out = next.getitem_next(nexthead, nexttail, nextcarry, 1, next.strides[0])
        return out.copy(shape=out.shape[1:], strides=out.strides[1:])

    def getitem_next(self, head, tail, carry, length, stride):
        print("getitem_next shape", self.shape, "strides", self.strides, "carry", carry, "length", length, "stride", stride)
        assert len(self.shape) == len(self.strides)

        if head is numpy.newaxis:
            raise NotImplementedError("numpy.newaxis")

        elif head is Ellipsis:
            raise NotImplementedError("...")

        elif isinstance(head, tuple) and len(head) == 0:
            print("shape", self.shape, "strides", self.strides, "carry", carry, "length", length, "stride", stride)

            ptr = numpy.full(len(carry)*stride, 123, dtype=numpy.uint8)
            for i in range(len(carry)):
                print("to", i*stride, ":", (i + 1)*stride, "from", self.byteoffset + carry[i]*stride, ":", self.byteoffset + (carry[i] + 1)*stride, "which is", self.ptr[self.byteoffset + carry[i]*stride : self.byteoffset + (carry[i] + 1)*stride])
                ptr[i*stride : (i + 1)*stride] = self.ptr[self.byteoffset + carry[i]*stride : self.byteoffset + (carry[i] + 1)*stride]

            print("ptr", ptr.view(self.dtype))

            shape = (len(carry),) + self.shape[1:]
            strides = (stride,) + self.strides[1:]
            return self.copy(ptr=ptr, shape=shape, strides=strides, byteoffset=0)

        elif isinstance(head, (int, numpy.integer)):
            raise NotImplementedError("int")

        elif isinstance(head, slice) and head.step is None:
            assert len(self.shape) > 1
            next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))

            nexthead, nexttail = head_tail(tail)
            nextcarry = numpy.full(len(carry)*(head.stop - head.start), 999, dtype=int)

            skip, remainder = divmod(self.strides[0], self.strides[1])
            assert remainder == 0
            for i in range(len(carry)):
                for j in range(head.stop - head.start):
                    nextcarry[i*(head.stop - head.start) + j] = skip*carry[i] + head.start + j

            out = next.getitem_next(nexthead, nexttail, nextcarry, length*(head.stop - head.start), next.strides[0])
            shape = (length, out.shape[0] // length) + out.shape[1:]   # maybe out.shape[0] // length == head.stop - head.start
            strides = (shape[1]*out.strides[0],) + out.strides
            return out.copy(shape=shape, strides=strides)

        elif isinstance(head, slice):
            raise NotImplementedError("slice3")

        else:
            head = numpy.asarray(head)
            if issubclass(head.dtype.type, numpy.integer):
                assert len(self.shape) != 0
                next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))

                nexthead, nexttail = head_tail(tail)
                nextcarry = numpy.full(len(carry)*len(head), 999, dtype=int)

                skip, remainder = divmod(self.strides[0], self.strides[1])
                assert remainder == 0
                for i in range(len(carry)):
                    for j in range(len(head)):
                        nextcarry[i*len(head) + j] = skip*carry[i] + head[j]

                out = next.getitem_next(nexthead, nexttail, nextcarry, length*len(head), next.strides[0])
                shape = (length, out.shape[0] // length) + out.shape[1:]   # maybe out.shape[0] // length == len(head)
                strides = (shape[1]*out.strides[0],) + out.strides
                return out.copy(shape=shape, strides=strides)

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

# a = numpy.arange(10)[::-1]
# b = NumpyArray(a)
# print(b.shape, b.strides, numpy.array(b))
# print("b.shape", b.shape, "b.strides", b.strides, "b.ptr", b.ptr)
# c = b.compact()
# print("c.shape", c.shape, "c.strides", c.strides, "c.ptr", c.ptr)
# print(a.tolist())
# print(c.tolist())
# assert c.iscompact
# if a.tolist() != c.tolist():
#     print("WRONG!!!")

a = numpy.arange(9*6).reshape(9, 6)[1::3, 1::2]
b = NumpyArray(a)
cut = (slice(1, 3), slice(1, 3))
acut = a[cut]
bcut = b[cut]
print("should be shape", acut.shape, "strides", acut.strides)
print("       is shape", bcut.shape, "strides", bcut.strides)
print(acut.tolist())
print(bcut.tolist())
if acut.tolist() != bcut.tolist():
    print("WRONG!!!")
