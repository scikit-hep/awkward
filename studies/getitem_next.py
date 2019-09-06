import numpy
import itertools
import functools
import operator

def condition1(self, tail):
    out = len(self.shape[1:]) == sum(1 if isinstance(x, int) else 0 for x in tail)
    return out

def condition2(self, tail):
    out1 = len(self.shape[1:]) == sum(1 if isinstance(x, int) else 0 for x in tail)
    out2 = any(isinstance(x, numpy.ndarray) for x in tail)
    return out1 or out2

def shape_product(x):
    return functools.reduce(operator.mul, x, 1)

def shape_deepsize(x):
    if len(x) < 2:
        return 1
    else:
        return shape_product(x[1:])

def shape_innersize(x):
    if len(x) < 2:
        return 1
    else:
        return x[1]

def shape_flatten(x):
    if len(x) < 2:
        return x
    else:
        return (x[0]*x[1],) + x[2:]

def shape_unflatten(x, by):
    return (by, x[0] // by) + x[1:]

def head_tail(x):
    head = () if len(x) == 0 else x[0]
    tail = x[1:]
    return head, tail

def spread_tail(tail, count):
    newtail = ()
    for unspread in tail:
        if not isinstance(unspread, numpy.ndarray):
            newtail = newtail + (unspread,)
        else:
            spread = numpy.full(len(unspread)*count, 999)
            k = 0
            for i in range(len(unspread)):
                for j in range(count):
                    spread[k] = unspread[i]
                    k += 1
            newtail = newtail + (spread,)
    return newtail

class NumpyArray:
    @classmethod
    def fromarray(cls, array):
        ptr = array.ravel()
        shape = array.shape
        offset = 0
        return cls(ptr, shape, offset)

    def __init__(self, ptr, shape, offset):
        self.ptr, self.shape, self.offset = ptr, shape, offset

    def tolist(self):
        if self.shape == ():
            return self.ptr[self.offset]
        else:
            return self.ptr[self.offset : self.offset + shape_product(self.shape)].reshape(self.shape).tolist()

    def __getitem__(self, where):
        if not isinstance(where, tuple):
            where = (where,)

        where = tuple(int(x[0]) if isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.integer) and x.shape == (1,) else x for x in where)

        head, tail = head_tail(where)
        return self.getitem_next(head, tail)

    def getitem_next(self, head, tail):
        if isinstance(head, tuple) and len(head) == 0:
            # print("null")
            return self

        elif isinstance(head, int):
            # print("int")
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")
            assert 0 <= head < self.shape[0]
            next = NumpyArray(self.ptr, self.shape[1:], self.offset + head*shape_product(self.shape[1:]))
            nexthead, nexttail = head_tail(tail)
            return next.getitem_next(nexthead, nexttail)

        elif isinstance(head, slice) and head.step is None:
            # print("slice2")
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")
            assert head.stop > head.start
            assert 0 <= head.start <  self.shape[0]
            assert 0 <  head.stop  <= self.shape[0]

            if len(tail) == 0:
                # an optimization: no carry needed if this is the last slice
                return NumpyArray(self.ptr, (head.stop - head.start,) + self.shape[1:], self.offset + head.start*shape_product(self.shape[1:]))

            innersize = shape_innersize(self.shape)
            nextshape = shape_flatten(self.shape)
            count = head.stop - head.start
            starts = numpy.full(count, 999)
            stops  = numpy.full(count, 999)
            for i in range(count):
                starts[i] = (i)*innersize
                stops[i]  = (i + 1)*innersize

            # print(" starts", starts)
            # print("   tail", tail)
            # newtail = spread_tail(tail, count)
            # print("newtail", newtail)
            nexthead, nexttail = head_tail(tail)   # newtail

            next = NumpyArray(self.ptr, nextshape, self.offset + head.start*shape_product(self.shape[1:]))
            out = next.getitem_next_carry(nexthead, nexttail, starts, stops, False)
            if condition1(self, tail):
                return out
            else:
                outshape = shape_unflatten(out.shape, len(starts))
                return NumpyArray(out.ptr, outshape, out.offset)

        elif isinstance(head, numpy.ndarray) and issubclass(head.dtype.type, numpy.integer) and len(head.shape) == 1:
            # print("array")
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")

            innersize = shape_innersize(self.shape)
            nextshape = shape_flatten(self.shape)
            count = len(head)
            starts = numpy.full(count, 999)
            stops  = numpy.full(count, 999)
            for i in range(count):
                assert 0 <= head[i] < self.shape[0]
                starts[i] = (head[i])*innersize
                stops[i]  = (head[i] + 1)*innersize

            next = NumpyArray(self.ptr, nextshape, self.offset)
            nexthead, nexttail = head_tail(tail)
            out = next.getitem_next_carry(nexthead, nexttail, starts, stops, True)
            if condition2(self, tail):
                return out
            else:
                outshape = shape_unflatten(out.shape, len(starts))
                return NumpyArray(out.ptr, outshape, out.offset)

        else:
            raise AssertionError

    def getitem_next_carry(self, head, tail, starts, stops, advanced):
        if isinstance(head, tuple) and len(head) == 0:
            # print("carry null", starts)
            deepsize = shape_deepsize(self.shape)
            length = sum((stop - start) for start, stop in zip(starts, stops))
            deeplength = deepsize*length
            ptr = numpy.full(deeplength, 999)

            where = 0
            for i in range(len(starts)):
                wherenext = where + deepsize*(stops[i] - starts[i])
                assert 0 <= where     <  len(ptr)
                assert 0 <  wherenext <= len(ptr)
                assert 0 <= self.offset + deepsize*starts[i] < len(self.ptr)

                ptr[where : wherenext] = self.ptr[self.offset + deepsize*starts[i] : self.offset + deepsize*stops[i]]
                where = wherenext

            return NumpyArray(ptr, (length,) + self.shape[1:], 0)

        elif isinstance(head, int):
            # print("carry int", starts)
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")
            innersize = shape_innersize(self.shape)
            nextstarts = numpy.full(len(starts), 999)
            nextstops  = numpy.full(len(starts), 999)
            for i in range(len(starts)):
                assert 0 <= head < stops[i] - starts[i]
                nextstarts[i] = (starts[i] + head)*innersize
                nextstops[i]  = (starts[i] + head + 1)*innersize

            next = NumpyArray(self.ptr, self.shape[1:], self.offset)
            nexthead, nexttail = head_tail(tail)
            return next.getitem_next_carry(nexthead, nexttail, nextstarts, nextstops, advanced)

        elif isinstance(head, slice) and head.step is None:
            # print("carry slice2", starts)
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")
            assert head.stop > head.start
            innersize = shape_innersize(self.shape)
            nextshape = shape_flatten(self.shape)
            count = head.stop - head.start
            nextstarts = numpy.full(len(starts)*count, 999)
            nextstops  = numpy.full(len(starts)*count, 999)
            k = 0
            for i in range(len(starts)):
                assert 0 <= head.start <  stops[i] - starts[i]
                assert 0 <  head.stop  <= stops[i] - starts[i]
                for j in range(count):
                    nextstarts[k] = (starts[i] + head.start + j)*innersize
                    nextstops[k]  = (starts[i] + head.start + j + 1)*innersize
                    k += 1

            # newtail = spread_tail(tail, count)
            nexthead, nexttail = head_tail(tail)   # newtail

            next = NumpyArray(self.ptr, nextshape, self.offset)
            out = next.getitem_next_carry(nexthead, nexttail, nextstarts, nextstops, advanced)
            if condition1(self, tail):
                return out
            else:
                outshape = shape_unflatten(out.shape, len(nextstarts))
                return NumpyArray(out.ptr, outshape, out.offset)

        elif isinstance(head, numpy.ndarray) and issubclass(head.dtype.type, numpy.integer) and len(head.shape) == 1:
            # print("carry array", starts)
            if len(self.shape) == 0:
                raise IndexError("too many indices for array")
            innersize = shape_innersize(self.shape)
            nextshape = shape_flatten(self.shape)
            if advanced:
                # print("starts", starts)
                # print("head  ", head)
                nextstarts = numpy.full(len(starts), 999)
                nextstops  = numpy.full(len(starts), 999)
                for i in range(len(starts)):
                    nextstarts[i] = (starts[i] + head[i])*innersize
                    nextstops[i]  = (starts[i] + head[i] + 1)*innersize

                # newtail = tail

            else:
                count = len(head)
                nextstarts = numpy.full(len(starts)*count, 999)
                nextstops  = numpy.full(len(starts)*count, 999)
                k = 0
                for i in range(len(starts)):
                    for j in range(count):
                        nextstarts[k] = (starts[i] + head[j])*innersize
                        nextstops[k]  = (starts[i] + head[j] + 1)*innersize
                        k += 1

                # newtail = spread_tail(tail, count)

            nexthead, nexttail = head_tail(tail)   # newtail

            next = NumpyArray(self.ptr, nextshape, self.offset)
            out = next.getitem_next_carry(nexthead, nexttail, nextstarts, nextstops, True)
            if condition2(self, tail):
                return out
            else:
                outshape = shape_unflatten(out.shape, len(nextstarts))
                return NumpyArray(out.ptr, outshape, out.offset)

        else:
            raise AssertionError

# a = numpy.arange(7*5).reshape(7, 5)
# a = numpy.arange(7*5*6).reshape(7, 5, 6)
a = numpy.arange(7*5*6*4).reshape(7, 5, 6, 4)
b = NumpyArray.fromarray(a)

# for depth in 1, 2:
#     for cuts in itertools.permutations((0, 1, slice(0, 5), slice(1, 4), slice(2, 3)), depth):
# for depth in 1, 2, 3:
#     for cuts in itertools.permutations((0, 1, 2, slice(0, 5), slice(1, 4), slice(2, 3)), depth):
for depth in 1, 2, 3, 4:
    for cuts in itertools.permutations((0, 1, 2, 3, slice(0, 4), slice(1, 3), slice(1, 2), slice(2, 3)), depth):
        print(cuts)
        acut = a[cuts].tolist()
        bcut = b[cuts].tolist()
        print(acut)
        print(bcut)
        print()
        assert acut == bcut

# a = numpy.arange(7*5).reshape(7, 5)
# b = NumpyArray.fromarray(a)
# print(a)
# cut = (numpy.array([0, 1]), 2)
# acut = a[cut]
# bcut = b[cut]
# print(acut.shape)
# print(bcut.shape)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

# cut = (slice(0, 2), numpy.array([0, 1, 2]), numpy.array([3, 4, 5]))
# a = numpy.arange(7*5*6).reshape(7, 5, 6)
# b = NumpyArray.fromarray(a)
# acut = a[cut]
# bcut = b[cut]
# print(acut.shape)
# print(bcut.shape)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")

# a = numpy.arange(7*5*6*4).reshape(7, 5, 6, 4)
# b = NumpyArray.fromarray(a)
# cut = (slice(1, 4), 2, slice(3, 4), 2)
# acut = a[cut]
# bcut = b[cut]
# print(acut.shape)
# print(bcut.shape)
# print(acut.tolist())
# print(bcut.tolist())
# if acut.tolist() != bcut.tolist():
#     print("WRONG!!!")
