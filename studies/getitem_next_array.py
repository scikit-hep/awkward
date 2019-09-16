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
        print("__init__ shape", self.shape, "strides", self.strides)

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
        if not isinstance(where, tuple):
            where = (where,)

        head, tail = head_tail(where)
        carry = numpy.array([0])
        return self.getitem_next(head, tail, carry)

    def getitem_next(self, head, tail, carry):
        print("getitem_next shape", self.shape, "strides", self.strides, "carry", carry)

        if head is numpy.newaxis:
            raise NotImplementedError("numpy.newaxis")

        elif head is Ellipsis:
            raise NotImplementedError("...")

        elif isinstance(head, tuple) and len(head) == 0:
            if len(self.strides) == 0:
                size = self.itemsize
            else:
                size = self.strides[0]

            ptr = numpy.full(len(carry)*size, 123, dtype=numpy.uint8)
            for i in range(len(carry)):
                ptr[i*size : (i + 1)*size] = self.ptr[self.byteoffset + carry[i]*size : self.byteoffset + (carry[i] + 1)*size]

            print("new ptr", ptr.view(self.dtype))

            return self.copy(ptr=self.ptr, shape=(len(carry),), strides=(self.itemsize,), byteoffset=0)

        elif isinstance(head, (int, numpy.integer)):
            raise NotImplementedError("int")

        elif isinstance(head, slice) and head.step is None:
            raise NotImplementedError("slice2")

        elif isinstance(head, slice):
            raise NotImplementedError("slice3")

        else:
            head = numpy.asarray(head)
            if issubclass(head.dtype.type, numpy.integer):
                next = self.copy(shape=flatten_shape(self.shape), strides=flatten_strides(self.strides))

                nexthead, nexttail = head_tail(tail)
                nextcarry = numpy.full(len(carry)*len(head), 999, dtype=int)
                for i in range(len(carry)):
                    for j in range(len(head)):
                        nextcarry[i*len(head) + j] = carry[i] + head[j]

                out = next.getitem_next(nexthead, nexttail, nextcarry)

                return out.copy(shape=unflatten_shape(len(head), self.shape), strides=unflatten_strides(len(head), self.strides, self.itemsize))

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

def unflatten_shape(length, shape):
    if len(shape) == 0:
        return (length,)
    else:
        return (length, shape[0] // length) + shape[1:]

def flatten_strides(strides):
    return strides[1:]

def unflatten_strides(length, strides, itemsize):
    if len(strides) == 0:
        return (itemsize,)
    else:
        return (length*strides[0],) + strides

a = numpy.arange(7).reshape(7)
b = NumpyArray(a)
print(a)
cut = (numpy.array([0]),)
acut = a[cut]
bcut = b[cut]
print("should be shape", acut.shape, "strides", acut.strides)
print("       is shape", bcut.shape, "strides", bcut.strides)
print(acut.tolist())
print(bcut.tolist())
if acut.tolist() != bcut.tolist():
    print("WRONG!!!")
