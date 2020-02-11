import math
import random

class Content:
    def __iter__(self):
        def convert(x):
            if isinstance(x, Content):
                return list(x)
            elif isinstance(x, tuple):
                return tuple(convert(y) for y in x)
            elif isinstance(x, dict):
                return {n: convert(y) for n, y in x.items()}
            else:
                return x

        for i in range(len(self)):
            yield convert(self[i])

    @staticmethod
    def random(minlen=0, choices=None):
        "Generate a random array from a set of possible classes."
        
        if choices is None:
            choices = [x for x in globals().values() if isinstance(x, type) and issubclass(x, Content)]
        else:
            choices = list(choices)
        if minlen != 0 and EmptyArray in choices:
            choices.remove(EmptyArray)
        assert len(choices) > 0
        cls = random.choice(choices)
        return cls.random(minlen, choices)

    def tolist(self):
        return list(self)

def random_number():
    return round(random.gauss(5, 3), 1)

def random_length(minlen=0, maxlen=None):
    if maxlen is None:
        return minlen + int(math.floor(random.expovariate(0.1)))
    else:
        return random.randint(minlen, maxlen)

class RawArray(Content):
    def __init__(self, ptr):
        assert isinstance(ptr, list)
        self.ptr = ptr

    @staticmethod
    def random(minlen=0, choices=None):
        return RawArray([random_number() for i in range(random_length(minlen))])

    def __len__(self):
        return len(self.ptr)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.ptr[where]
        elif isinstance(where, slice) and where.step is None:
            return RawArray(self.ptr[where])
        elif isinstance(where, str):
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "RawArray(" + repr(self.ptr) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<RawArray>\n"
        out += indent + "    <ptr>" + " ".join(repr(x) for x in self.ptr) + "</ptr>\n"
        out += indent + "</RawArray>" + post
        return out

class NumpyArray(Content):
    def __init__(self, ptr, shape, strides, offset):
        assert isinstance(ptr, list)
        assert isinstance(shape, list)
        assert isinstance(strides, list)
        for x in ptr:
            assert isinstance(x, (bool, int, float))
        assert len(shape) > 0
        assert len(strides) == len(shape)
        for x in shape:
            assert isinstance(x, int)
            assert x >= 0
        for x in strides:
            assert isinstance(x, int)
        assert isinstance(offset, int)
        if all(x != 0 for x in shape):
            assert 0 <= offset < len(ptr)
            last = offset
            for sh, st in zip(shape, strides):
                last += (sh - 1) * st
            assert last <= len(ptr)
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.offset = offset

    @staticmethod
    def random(minlen=0, choices=None):
        shape = [random_length(minlen)]
        for i in range(random_length(0, 2)):
            shape.append(random_length(1, 3))
        strides = [1]
        for x in shape[:0:-1]:
            skip = random_length(0, 2)
            strides.insert(0, x * strides[0] + skip)
        offset = random_length()
        ptr = [random_number() for i in range(shape[0] * strides[0] + offset)]
        return NumpyArray(ptr, shape, strides, offset)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            offset = self.offset + self.strides[0] * where
            if len(self.shape) == 1:
                return self.ptr[offset]
            else:
                return NumpyArray(self.ptr, self.shape[1:], self.strides[1:], offset)
        elif isinstance(where, slice) and where.step is None:
            offset = self.offset + self.strides[0] * where.start
            shape = [where.stop - where.start] + self.shape[1:]
            return NumpyArray(self.ptr, shape, self.strides, offset)
        elif isinstance(where, str):
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "NumpyArray(" + repr(self.ptr) + ", " + repr(self.shape) + ", " + repr(self.strides) + ", " + repr(self.offset) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<NumpyArray>\n"
        out += indent + "    <ptr>" + " ".join(str(x) for x in self.ptr) + "</ptr>\n"
        out += indent + "    <shape>" + " ".join(str(x) for x in self.shape) + "</shape>\n"
        out += indent + "    <strides>" + " ".join(str(x) for x in self.strides) + "</strides>\n"
        out += indent + "    <offset>" + str(self.offset) + "</offset>\n"
        out += indent + "</NumpyArray>" + post
        return out

class EmptyArray(Content):
    def __init__(self):
        pass

    @staticmethod
    def random(minlen=0, choices=None):
        assert minlen == 0
        return EmptyArray()

    def __len__(self):
        return 0

    def __getitem__(self, where):
        if isinstance(where, int):
            assert False
        elif isinstance(where, slice) and where.step is None:
            return EmptyArray()
        elif isinstance(where, str):
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "EmptyArray()"

    def toxml(self, indent="", pre="", post=""):
        return indent + pre + "<EmptyArray/>" + post

class RegularArray(Content):
    def __init__(self, content, size):
        assert isinstance(content, Content)
        assert isinstance(size, int)
        assert size > 0
        self.content = content
        self.size = size

    @staticmethod
    def random(minlen=0, choices=None):
        size = random_length(1, 5)
        return RegularArray(Content.random(random_length(minlen) * size, choices), size)

    def __len__(self):
        return len(self.content) // self.size   # floor division

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.content[(where) * self.size:(where + 1) * self.size]
        elif isinstance(where, slice) and where.step is None:
            start = where.start * self.size
            stop = where.stop * self.size
            return RegularArray(self.content[start:stop], self.size)
        elif isinstance(where, str):
            return RegularArray(self.content[where], self.size)
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "RegularArray(" + repr(self.content) + ", " + repr(self.size) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<RegularArray>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "    <size>" + str(self.size) + "</size>\n"
        out += indent + "</RegularArray>" + post
        return out

class ListOffsetArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Content)
        assert len(offsets) != 0
        for i in range(len(offsets) - 1):
            start = offsets[i]
            stop = offsets[i + 1]
            assert isinstance(start, int)
            assert isinstance(stop, int)
            if start != stop:
                assert start < stop   # i.e. start <= stop
                assert start >= 0
                assert stop <= len(content)
        self.offsets = offsets
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        counts = [random_length() for i in range(random_length(minlen))]
        offsets = [random_length()]
        for x in counts:
            offsets.append(offsets[-1] + x)
        return ListOffsetArray(offsets, Content.random(offsets[-1], choices))
        
    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.content[self.offsets[where]:self.offsets[where + 1]]
        elif isinstance(where, slice) and where.step is None:
            offsets = self.offsets[where.start : where.stop + 1]
            if len(offsets) == 0:
                offsets = [0]
            return ListOffsetArray(offsets, self.content)
        elif isinstance(where, str):
            return ListOffsetArray(self.offsets, self.content[where])
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "ListOffsetArray(" + repr(self.offsets) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<ListOffsetArray>\n"
        out += indent + "    <offsets>" + " ".join(str(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListOffsetArray>" + post
        return out

class ListArray(Content):
    def __init__(self, starts, stops, content):
        assert isinstance(starts, list)
        assert isinstance(stops, list)
        assert isinstance(content, Content)
        assert len(stops) >= len(starts)   # usually equal
        for i in range(len(starts)):
            start = starts[i]
            stop = stops[i]
            assert isinstance(start, int)
            assert isinstance(stop, int)
            if start != stop:
                assert start < stop   # i.e. start <= stop
                assert start >= 0
                assert stop <= len(content)
        self.starts = starts
        self.stops = stops
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        content = Content.random(0, choices)
        length = random_length(minlen)
        if len(content) == 0:
            starts = [random.randint(0, 10) for i in range(length)]
            stops = list(starts)
        else:
            starts = [random.randint(0, len(content) - 1) for i in range(length)]
            stops = [x + min(random_length(), len(content) - x) for x in starts]
        return ListArray(starts, stops, content)
        
    def __len__(self):
        return len(self.starts)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.content[self.starts[where]:self.stops[where]]
        elif isinstance(where, slice) and where.step is None:
            starts = self.starts[where.start:where.stop]
            stops = self.stops[where.start:where.stop]
            return ListArray(starts, stops, self.content)
        elif isinstance(where, str):
            return ListArray(self.starts, self.stops, self.content[where])
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "ListArray(" + repr(self.starts) + ", " + repr(self.stops) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<ListArray>\n"
        out += indent + "    <starts>" + " ".join(str(x) for x in self.starts) + "</starts>\n"
        out += indent + "    <stops>" + " ".join(str(x) for x in self.stops) + "</stops>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListArray>" + post
        return out

class IndexedArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, list)
        assert isinstance(content, Content)
        for x in index:
            assert isinstance(x, int)
            assert 0 <= x < len(content)   # index[i] must not be negative
        self.index = index
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        if minlen == 0:
            content = Content.random(0, choices)
        else:
            content = Content.random(1, choices)
        if len(content) == 0:
            index = []
        else:
            index = [random.randint(0, len(content) - 1) for i in range(random_length(minlen))]
        return IndexedArray(index, content)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.content[self.index[where]]
        elif isinstance(where, slice) and where.step is None:
            return IndexedArray(self.index[where.start:where.stop], self.content)
        elif isinstance(where, str):
            return IndexedArray(self.index, self.content[where])
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "IndexedArray(" + repr(self.index) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<IndexedArray>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</IndexedArray>\n"
        return out

class IndexedOptionArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, list)
        assert isinstance(content, Content)
        for x in index:
            assert isinstance(x, int)
            assert x < len(content)   # index[i] may be negative
        self.index = index
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        content = Content.random(0, choices)
        index = []
        for i in range(random_length(minlen)):
            if len(content) == 0 or random.randint(0, 4) == 0:
                index.append(-random_length(1))   # a random number, but not necessarily -1
            else:
                index.append(random.randint(0, len(content) - 1))
        return IndexedOptionArray(index, content)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            if self.index[where] < 0:
                return None
            else:
                return self.content[self.index[where]]
        elif isinstance(where, slice) and where.step is None:
            return IndexedOptionArray(self.index[where.start:where.stop], self.content)
        elif isinstance(where, str):
            return IndexedOptionArray(self.index, self.content[where])
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "IndexedOptionArray(" + repr(self.index) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<IndexedOptionArray>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</IndexedOptionArray>\n"
        return out

class ByteMaskedArray(Content):
    def __init__(self, mask, content, validwhen):
        assert isinstance(mask, list)
        assert isinstance(content, Content)
        assert isinstance(validwhen, bool)
        assert len(mask) <= len(content)
        for x in mask:
            assert isinstance(x, bool)
        self.mask = mask
        self.content = content
        self.validwhen = validwhen

    @staticmethod
    def random(minlen=0, choices=None):
        validwhen = random.choice([False, True])
        mask = [(random.randint(0, 4) == 0) ^ validwhen for i in range(random_length(minlen))]   # 80% are valid
        content = Content.random(len(mask), choices)
        return ByteMaskedArray(mask, content, validwhen)

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            if self.mask[where] == self.validwhen:
                return self.content[where]
            else:
                return None
        elif isinstance(where, slice) and where.step is None:
            return ByteMaskedArray(self.mask[where.start:where.stop], self.content[where.start:where.stop], self.validwhen)
        elif isinstance(where, str):
            return ByteMaskedArray(self.mask, self.content[where], self.validwhen)
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "ByteMaskedArray(" + repr(self.mask) + ", " + repr(self.content) + ", " + repr(self.validwhen) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<ByteMaskedArray validwhen=\"" + repr(self.validwhen) + "\">\n"
        out += indent + "    <mask>" + " ".join(str(x) for x in self.mask) + "</mask>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ByteMaskedArray>\n"
        return out

class RecordArray(Content):
    def __init__(self, contents, recordlookup, length):
        assert isinstance(contents, list)
        if len(contents) == 0:
            assert isinstance(length, int)
            assert length >= 0
        else:
            assert length is None
            for x in contents:
                assert isinstance(x, Content)
        assert recordlookup is None or isinstance(recordlookup, list)
        if isinstance(recordlookup, list):
            assert len(recordlookup) == len(contents)
            for x in recordlookup:
                assert isinstance(x, str)
        self.contents = contents
        self.recordlookup = recordlookup
        self.length = length

    @staticmethod
    def random(minlen=0, choices=None):
        length = random_length(minlen)
        contents = []
        for i in range(random.randint(0, 2)):
            contents.append(Content.random(length, choices))
        if len(contents) != 0:
            length = None
        if random.randint(0, 1) == 0:
            recordlookup = None
        else:
            recordlookup = ["x" + str(i) for i in range(len(contents))]
        return RecordArray(contents, recordlookup, length)

    def __len__(self):
        if len(self.contents) == 0:
            return self.length
        else:
            return min(len(x) for x in self.contents)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            record = [x[where] for x in self.contents]
            if self.recordlookup is None:
                return tuple(record)
            else:
                return dict(zip(self.recordlookup, record))
        elif isinstance(where, slice) and where.step is None:
            if len(self.contents) == 0:
                start = min(max(where.start, 0), self.length)
                stop = min(max(where.stop, 0), self.length)
                if stop < start:
                    stop = start
                return RecordArray([], self.recordlookup, stop - start)
            else:
                return RecordArray([x[where] for x in self.contents], self.recordlookup, self.length)
        elif isinstance(where, str):
            if self.recordlookup is None:
                try:
                    i = int(where)
                except ValueError:
                    pass
                else:
                    if i < len(self.contents):
                        return self.contents[i]
            else:
                try:
                    i = self.recordlookup.index(where)
                except ValueError:
                    pass
                else:
                    return self.contents[i]
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "RecordArray([" + ", ".join(repr(x) for x in self.contents) + "], " + repr(self.recordlookup) + ", " + repr(self.length) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<RecordArray>\n"
        if len(self.contents) == 0:
            out += indent + "    <istuple>" + str(self.recordlookup is None) + "</istuple>\n"
            out += indent + "    <length>" + str(self.length) + "</length>\n"
        if self.recordlookup is None:
            for i, content in enumerate(self.contents):
                out += content.toxml(indent + "    ", "<content i=\"" + str(i) + "\">", "</content>\n")
        else:
            for i, (key, content) in enumerate(zip(self.recordlookup, self.contents)):
                out += content.toxml(indent + "    ", "<content i=\"" + str(i) + "\" key=\"" + repr(key) + "\">", "</content>\n")
        out += indent + "</RecordArray>" + post
        return out

class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, list)
        assert isinstance(index, list)
        assert isinstance(contents, list)
        assert len(index) >= len(tags)   # usually equal
        for x in tags:
            assert isinstance(x, int)
            assert 0 <= x < len(contents)
        for i, x in enumerate(index):
            assert isinstance(x, int)
            assert 0 <= x < len(contents[tags[i]])
        self.tags = tags
        self.index = index
        self.contents = contents

    @staticmethod
    def random(minlen=0, choices=None):
        contents = []
        unshuffled_tags = []
        unshuffled_index = []
        for i in range(random.randint(1, 3)):
            if minlen == 0:
                contents.append(Content.random(0, choices))
            else:
                contents.append(Content.random(1, choices))
            if len(contents[-1]) != 0:
                thisindex = [random.randint(0, len(contents[-1]) - 1) for i in range(random_length(minlen))]
                unshuffled_tags.extend([i] * len(thisindex))
                unshuffled_index.extend(thisindex)
        permutation = list(range(len(unshuffled_tags)))
        random.shuffle(permutation)
        tags = [unshuffled_tags[i] for i in permutation]
        index = [unshuffled_index[i] for i in permutation]
        return UnionArray(tags, index, contents)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.contents[self.tags[where]][self.index[where]]
        elif isinstance(where, slice) and where.step is None:
            return UnionArray(self.tags[where], self.index[where], self.contents)
        elif isinstance(where, str):
            return UnionArray(self.tags, self.index, [x[where] for x in self.contents])
        else:
            raise AssertionError(where)

    def __repr__(self):
        return "UnionArray(" + repr(self.tags) + ", " + repr(self.index) + ", [" + ", ".join(repr(x) for x in self.contents) + "])"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<UnionArray>\n"
        out += indent + "    <tags>" + " ".join(str(x) for x in self.tags) + "</tags>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        for i, content in enumerate(self.contents):
            out += content.toxml(indent + "    ", "<content i=\"" + str(i) + "\">", "</content>\n")
        out += indent + "</UnionArray>" + post
        return out

###################################################################### use NumPy to define correct behavior

import numpy

# (My example reducer will be 'prod' on prime numbers; can't get the right answer accidentally.)
primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]

nparray = numpy.array(primes[:2*3*5]).reshape(2, 3, 5)

assert (nparray.tolist() ==
    [[[  2,   3,   5,   7,  11],
      [ 13,  17,  19,  23,  29],
      [ 31,  37,  41,  43,  47]],
     [[ 53,  59,  61,  67,  71],
      [ 73,  79,  83,  89,  97],
      [101, 103, 107, 109, 113]]])
assert nparray.shape == (2, 3, 5)

assert (numpy.prod(nparray, axis=0).tolist() ==
    [[ 106,  177,  305,  469, 781],
     [ 949, 1343, 1577, 2047, 2813],
     [3131, 3811, 4387, 4687, 5311]])
assert numpy.prod(nparray, axis=0).shape == (3, 5)

assert (numpy.prod(nparray, axis=1).tolist() ==
    [[   806,   1887,   3895,   6923, 14993],
     [390769, 480083, 541741, 649967, 778231]])
assert numpy.prod(nparray, axis=1).shape == (2, 5)

assert (numpy.prod(nparray, axis=2).tolist() ==
    [[     2310,    2800733,    95041567],
     [907383479, 4132280413, 13710311357]])
assert numpy.prod(nparray, axis=2).shape == (2, 3)

# (axis=3 is a numpy.AxisError; axis<0 counts backward from the end.)

assert numpy.prod(nparray, axis=None) == 7581744426003940878
# (but that should be a special case because it results in a scalar.)

akarray = NumpyArray(primes[:2*3*5], [2, 3, 5], [15, 5, 1], 0)

assert (akarray.tolist() ==
    [[[  2,   3,   5,   7,  11],
      [ 13,  17,  19,  23,  29],
      [ 31,  37,  41,  43,  47]],
     [[ 53,  59,  61,  67,  71],
      [ 73,  79,  83,  89,  97],
      [101, 103, 107, 109, 113]]])
assert akarray.shape == [2, 3, 5]

def NumpyArray_prod(self, axis):
    assert len(self.shape) != 0

    if axis < 0:
        axis += len(self.shape)
    assert 0 <= axis < len(self.shape)

    if axis == 0:
        length = self.shape[0]
        stepsize = self.strides[0]

        shape = self.shape[1:]
        strides = [1]
        for x in shape[:0:-1]:
            strides = [strides[0] * x] + strides

        flatlen = 1
        for x in shape:
            flatlen *= x

        index = [None] * (flatlen*self.shape[0])
        parents = [None] * (flatlen*self.shape[0])

        def recurse(k, n, base):
            if n < len(shape) - 1:
                for i in range(shape[n]):
                    k = recurse(k, n + 1, base + i*self.strides[n + 1])   # item-strides, not byte-strides
            elif n == len(shape) - 1:
                for i in range(shape[n]):
                    for j in range(self.shape[0]):
                        index[k*self.shape[0] + j] = base + j*self.strides[0] + i
                        parents[k*self.shape[0] + j] = k
                    k += 1
            return k

        recurse(0, 0, self.offset)   # item-offset, not byte-offset

        ptr = [1] * flatlen   # initialize to identity because lists can be empty

        lastparent = -1
        for i in range(flatlen*self.shape[0]):
            if parents[i] != lastparent:
                ptr[parents[i]] = self.ptr[index[i]]
            else:
                ptr[parents[i]] = self.ptr[index[i]] * ptr[lastparent]
            lastparent = parents[i]

        return NumpyArray(ptr, shape, strides, 0)

NumpyArray.prod = NumpyArray_prod

assert (akarray.prod(axis=0).tolist() ==
    [[ 106,  177,  305,  469, 781],
     [ 949, 1343, 1577, 2047, 2813],
     [3131, 3811, 4387, 4687, 5311]])
assert akarray.prod(axis=0).shape == [3, 5]
