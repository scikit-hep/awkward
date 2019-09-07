import numpy
import itertools
import functools
import operator

class NumpyArray:
    def __init__(self, array):
        self.ptr = array.tostring()
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

    def __getitem__(self, where):
        if not isinstance(where, tuple):
            where = (where,)

        where = tuple(int(x[0]) if isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.integer) and x.shape == (1,) else x for x in where)

        head, tail = head_tail(where)
        return self.getitem_next(head, tail, 1)

    def getitem_next(self, head, tail, repetition):
        if isinstance(head, tuple) and len(head) == 0:
            print("null")
            return self

        elif isinstance(head, int):
            print("int")
            raise NotImplementedError

        elif isinstance(head, slice) and head.step is None:
            print("slice2")
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")

            assert head.stop > head.start
            assert 0 <= head.start <  self.shape[0]
            assert 0 <  head.stop  <= self.shape[0]

            nextshape = (head.stop - head.start,) + self.shape[1:]
            nextbyteoffset = self.byteoffset + head.start*shape_product(self.shape[1:])*self.itemsize

            next = self.copy(shape=flatten(nextshape), strides=self.strides[1:], byteoffset=nextbyteoffset)

            nexthead, nexttail = head_tail(tail)
            out = next.getitem_next(nexthead, nexttail, repetition*nextshape[0])

            if isinstance(nexthead, tuple) and len(nexthead) == 0:
                toshape = nextshape
            elif self.ptr is out.ptr:
                toshape = (nextshape[0],) + out.shape
            else:
                toshape = unflatten(head.stop - head.start, out.shape)

            if len(out.strides) == 0:
                tostrides = (out.itemsize,)
            elif self.ptr is out.ptr:
                tostrides = (self.strides[0],) + out.strides
            else:
                tostrides = (out.strides[0]*out.shape[0] // nextshape[0],) + out.strides

            return out.copy(shape=toshape, strides=tostrides)

        elif isinstance(head, numpy.ndarray) and issubclass(head.dtype.type, numpy.integer) and len(head.shape) == 1:
            print("intarray")
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")

            copylen  = shape_product(self.shape[1:])*self.itemsize
            copyfrom = numpy.frombuffer(self.ptr, dtype=numpy.uint8)
            copyto   = numpy.zeros(repetition*len(head)*copylen, dtype=numpy.uint8)
            skip     = self.shape[0] // repetition

            for rep in range(repetition):
                for i in range(len(head)):
                    copyto[(rep*len(head) + i)*copylen : (rep*len(head) + i + 1)*copylen] = copyfrom[self.byteoffset + (rep*skip + head[i])*copylen : self.byteoffset + (rep*skip + head[i] + 1)*copylen]

            nextshape = (repetition*len(head),) + self.shape[1:]

            # copyto.tostring()
            next = self.copy(ptr=copyto, shape=flatten(nextshape), strides=self.strides[1:], byteoffset=0)

            nexthead, nexttail = head_tail(tail)
            out = next.getitem_next(nexthead, nexttail, repetition*len(head))

            if len(tail) == 0:
                toshape = nextshape
            else:
                toshape = (nextshape[0],) + out.shape

            tostrides = (self.strides[0],) + out.strides

            tmp = out.copy(shape=toshape, strides=tostrides)

            print("tmp", tmp.tolist())

            return tmp

        else:
            raise AssertionError

def head_tail(x):
    head = () if len(x) == 0 else x[0]
    tail = x[1:]
    return head, tail

def flatten(x):
    if len(x) == 1:
        return ()
    else:
        return (x[0]*x[1],) + x[2:]

def unflatten(by, x):
    return (by, x[0] // by) + x[1:]

def shape_product(x):
    return functools.reduce(operator.mul, x, 1)

def shape_innersize(x):
    if len(x) < 2:
        return 1
    else:
        return x[1]

# a = numpy.arange(7*5).reshape(7, 5)
# b = NumpyArray(a)
# print(a)
# cut = (numpy.array([4, 3, 1]), slice(1, 4))
# acut = a[cut]
# bcut = b[cut]
# print(acut.shape, acut.strides)
# print(bcut.shape, bcut.strides)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

# cut = (slice(2, 3), slice(1, 3), slice(2, 5))
# a = numpy.arange(7*5*6).reshape(7, 5, 6)
# b = NumpyArray(a)
# acut = a[cut]
# bcut = b[cut]
# print(acut.shape)
# print(bcut.shape)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

a = numpy.arange(7*5*6*4).reshape(7, 5, 6, 4)
b = NumpyArray(a)
cut = (slice(1, 3), slice(1, 3), numpy.array([0, 1]))
acut = a[cut]
bcut = b[cut]
print(acut.shape, acut.strides)
print(bcut.shape, bcut.strides)
print(acut.tolist())
print(bcut.tolist())
if acut.tolist() != bcut.tolist():
    print("WRONG!!!")

# # a = numpy.arange(7*5).reshape(7, 5)
# # a = numpy.arange(7*5*6).reshape(7, 5, 6)
# a = numpy.arange(7*5*6*4).reshape(7, 5, 6, 4)
# b = NumpyArray(a)

# # for depth in 1, 2:
# #     for cuts in itertools.permutations((slice(0, 5), slice(1, 4), slice(2, 3)), depth):
# # for depth in 1, 2, 3:
# #     for cuts in itertools.permutations((slice(0, 5), slice(1, 4), slice(2, 3)), depth):
# for depth in 1, 2, 3, 4:
#     for cuts in itertools.permutations((slice(0, 4), slice(1, 3), slice(1, 2), slice(2, 3)), depth):
#         print(cuts)
#         acut = a[cuts].tolist()
#         bcut = b[cuts].tolist()
#         print(acut)
#         print(bcut)
#         print()
#         assert acut == bcut
