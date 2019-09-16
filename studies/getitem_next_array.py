import numpy
import itertools
import functools
import operator

class NumpyArray:
    def __init__(self, array):
        self.ptr = numpy.frombuffer(array.tostring(), dtype=numpy.uint8)
        self.shape = array.shape
        self.strides = array.strides
        self.itemsize = array.itemsize
        self.dtype = array.dtype
        self.byteoffset = 0

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
            ptr = self.ptr[self.byteoffset : self.byteoffset + self.strides[0]*self.shape[0]]
            return numpy.lib.stride_tricks.as_strided(numpy.frombuffer(ptr, dtype=self.dtype), self.shape, self.strides)
            return out

    def tolist(self):
        return numpy.array(self).tolist()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, where):
        assert len(self.shape) != 0

        if not isinstance(where, tuple):
            where = (where,)

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
            ptr = numpy.full(len(carry)*stride, 123, dtype=numpy.uint8)
            for i in range(len(carry)):
                ptr[i*stride : (i + 1)*stride] = self.ptr[self.byteoffset + carry[i]*stride : self.byteoffset + (carry[i] + 1)*stride]
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
            for i in range(len(carry)):
                for j in range(head.stop - head.start):
                    nextcarry[i*(head.stop - head.start) + j] = self.shape[1]*carry[i] + head.start + j

            out = next.getitem_next(nexthead, nexttail, nextcarry, length*(head.stop - head.start), next.strides[0])
            shape = (length, out.shape[0] // length) + out.shape[1:]
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
                for i in range(len(carry)):
                    for j in range(len(head)):
                        nextcarry[i*len(head) + j] = carry[i] + head[j]

                return next.getitem_next(nexthead, nexttail, nextcarry, self.shape[0], self.strides[0])

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

a = numpy.arange(7*5*6).reshape(7, 5, 6)
b = NumpyArray(a)
cut = (slice(1, 5), slice(1, 4), slice(1, 3))
acut = a[cut]
bcut = b[cut]
print("should be shape", acut.shape, "strides", acut.strides)
print("       is shape", bcut.shape, "strides", bcut.strides)
print(acut.tolist())
print(bcut.tolist())
if acut.tolist() != bcut.tolist():
    print("WRONG!!!")
