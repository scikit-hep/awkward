import numpy
import itertools
import functools
import operator

def shape_product(x):
    return functools.reduce(operator.mul, x, 1)

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

def head_tail(x):
    head = () if len(x) == 0 else x[0]
    tail = x[1:]
    return head, tail

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
        head, tail = head_tail(where)
        return self.getitem_next(head, tail)

    def getitem_next(self, head, tail):
        if head == ():
            return self

        elif isinstance(head, int):
            shape = self.shape[1:]
            offset = self.offset + shape_product(self.shape[1:])*head
            nexthead, nexttail = head_tail(tail)
            return NumpyArray(self.ptr, shape, offset).getitem_next(nexthead, nexttail)

        elif isinstance(head, slice) and head.step is None:
            length = head.stop - head.start
            offset = self.offset + shape_product(self.shape[1:])*head.start
            nexthead, nexttail = head_tail(tail)

            if nexthead == ():
                # this is correct for slice2 -> ()
                shape = self.shape[1:]
                next = NumpyArray(self.ptr, shape, offset).getitem_next(nexthead, nexttail)
                return NumpyArray(next.ptr, (length,) + next.shape, next.offset)

            elif isinstance(nexthead, int):
                # this is correct for slice2 -> int
                shape = (length*self.shape[1],) + self.shape[2:]
                nextcarry = [i*self.shape[1] for i in range(length)]
                return NumpyArray(self.ptr, shape, offset).getitem_next_carry(nexthead, nexttail, nextcarry)

            else:
                # this is correct for slice2 -> slice2
                shape = (length*self.shape[1],) + self.shape[2:]
                nextcarry = [i*self.shape[1] for i in range(length)]
                next = NumpyArray(self.ptr, shape, offset).getitem_next_carry(nexthead, nexttail, nextcarry)
                outshape = (length, next.shape[0] // length) + self.shape[2:]
                return NumpyArray(next.ptr, outshape, next.offset)

        else:
            raise AssertionError

    def getitem_next_carry(self, head, tail, carry):
        if head == ():
            # this is correct for slice2 -> int
            # chunksize = shape_product(self.shape[1:])
            # ptr = numpy.full(len(carry)*chunksize, 999)
            # for i, x in enumerate(carry):
            #     ptr[i*chunksize : (i + 1)*chunksize] = self.ptr[self.offset + x*chunksize : self.offset + (x + 1)*chunksize]
            # return NumpyArray(ptr, (len(carry),) + self.shape[1:], 0)

            # this is correct for slice2 -> int -> int

            print("head ", head)
            print("self ", self.tolist())
            print("carry", carry)
            print("this ", numpy.array(self.tolist())[carry])

            print("shape", self.shape)
            chunksize = shape_product(self.shape[1:])
            print("chunksize", chunksize)

            ptr = numpy.full(len(carry)*chunksize, 999)
            for i, x in enumerate(carry):
                ptr[i*chunksize : (i + 1)*chunksize] = self.ptr[self.offset + x*chunksize : self.offset + (x + 1)*chunksize]
            return NumpyArray(ptr, (len(carry),) + self.shape[1:], 0)

        elif isinstance(head, int):
            # this is correct for slice2 -> int
            # nexthead, nexttail = head_tail(tail)
            # nextcarry = [x + head for x in carry]
            # return self.getitem_next_carry(nexthead, nexttail, nextcarry)

            # this is correct for slice2 -> int -> int
            nexthead, nexttail = head_tail(tail)
            innersize = shape_innersize(self.shape)
            nextcarry = [(x + head)*innersize for x in carry]
            nextshape = shape_flatten(self.shape)
            next = NumpyArray(self.ptr, nextshape, self.offset)
            return next.getitem_next_carry(nexthead, nexttail, nextcarry)

        elif isinstance(head, slice) and head.step is None:
            length = head.stop - head.start
            nextcarry = [999]*length*len(carry)
            k = 0
            for i in range(len(carry)):
                for j in range(head.start, head.stop):
                    nextcarry[k] = carry[i] + j
                    k += 1
            nexthead, nexttail = head_tail(tail)
            return self.getitem_next_carry(nexthead, nexttail, nextcarry)

        else:
            raise AssertionError

# a = numpy.arange(7*5).reshape(7, 5)
a = numpy.arange(7*5*6*4).reshape(7, 5, 6, 4)
# a = numpy.arange(7*5*6).reshape(6, 7, 5, 6)
b = NumpyArray.fromarray(a)

print(a[slice(2, 5), 1, 2].tolist())
print(b[slice(2, 5), 1, 2].tolist())





# for depth in 1, 2:
#     for cuts in itertools.permutations((0, 1, 2, slice(0, 2), slice(1, 3), slice(1, 4)), depth):
#         print(cuts)
#         acut = a[cuts].tolist()
#         bcut = b[cuts].tolist()
#         print(acut)
#         print(bcut)
#         print()
#         assert acut == bcut

