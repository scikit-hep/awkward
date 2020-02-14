import math
import random

################################################################ Content

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

    def __repr__(self):
        return self.tostring_part("", "", "").rstrip()

    @staticmethod
    def random(minlen=0, choices=None):
        if choices is None:
            choices = [x for x in globals().values() if isinstance(x, type) and issubclass(x, Content)]
        else:
            choices = list(choices)
        if minlen != 0 and EmptyArray in choices:
            choices.remove(EmptyArray)
        assert len(choices) > 0
        cls = random.choice(choices)
        return cls.random(minlen, choices)

def random_number():
    return round(random.gauss(5, 3), 1)

def random_length(minlen=0, maxlen=None):
    if maxlen is None:
        return minlen + int(math.floor(random.expovariate(0.1)))
    else:
        return random.randint(minlen, maxlen)

################################################################ RawArray

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<RawArray>\n"
        out += indent + "    <ptr>" + " ".join(repr(x) for x in self.ptr) + "</ptr>\n"
        out += indent + "</RawArray>" + post
        return out

    def constructor(self):
        return "RawArray(" + repr(self.ptr) + ")"

################################################################ NumpyArray

class NumpyArray(Content):
    def __init__(self, ptr, shape, strides, offset):
        # ignoring the distinction between bytes and elements; i.e. itemsize == 1
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
            # strides can be negative or zero!
        assert isinstance(offset, int)
        if all(x != 0 for x in shape):
            assert 0 <= offset < len(ptr)
            assert shape[0] * strides[0] + offset <= len(ptr)
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.offset = offset

    @classmethod
    def onedim(cls, data):
        return cls(data, [len(data)], [1], 0)

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<NumpyArray>\n"
        out += indent + "    <ptr>" + " ".join(str(x) for x in self.ptr) + "</ptr>\n"
        out += indent + "    <shape>" + " ".join(str(x) for x in self.shape) + "</shape>\n"
        out += indent + "    <strides>" + " ".join(str(x) for x in self.strides) + "</strides>\n"
        out += indent + "    <offset>" + str(self.offset) + "</offset>\n"
        out += indent + "</NumpyArray>" + post
        return out

    def constructor(self):
        return "NumpyArray(" + repr(self.ptr) + ", " + repr(self.shape) + ", " + repr(self.strides) + ", " + repr(self.offset) + ")"

################################################################ EmptyArray

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        return indent + pre + "<EmptyArray/>" + post

    def constructor(self):
        return "EmptyArray()"

################################################################ RegularArray

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<RegularArray>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "    <size>" + str(self.size) + "</size>\n"
        out += indent + "</RegularArray>" + post
        return out

    def constructor(self):
        return "RegularArray(" + self.content.constructor() + ", " + repr(self.size) + ")"

################################################################ ListArray

class ListArray(Content):
    def __init__(self, starts, stops, content):
        assert isinstance(starts, list)
        assert isinstance(stops, list)
        assert isinstance(content, Content)
        assert len(stops) >= len(starts)   # usually ==
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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<ListArray>\n"
        out += indent + "    <starts>" + " ".join(str(x) for x in self.starts) + "</starts>\n"
        out += indent + "    <stops>" + " ".join(str(x) for x in self.stops) + "</stops>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListArray>" + post
        return out

    def constructor(self):
        return "ListArray(" + repr(self.starts) + ", " + repr(self.stops) + ", " + self.content.constructor() + ")"

################################################################ ListOffsetArray

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<ListOffsetArray>\n"
        out += indent + "    <offsets>" + " ".join(str(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListOffsetArray>" + post
        return out

    def constructor(self):
        return "ListOffsetArray(" + repr(self.offsets) + ", " + self.content.constructor() + ")"

################################################################ IndexedArray

class IndexedArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, list)
        assert isinstance(content, Content)
        for x in index:
            assert isinstance(x, int)
            assert 0 <= x < len(content)   # index[i] may not be negative
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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<IndexedArray>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</IndexedArray>\n"
        return out

    def constructor(self):
        return "IndexedArray(" + repr(self.index) + ", " + self.content.constructor() + ")"

################################################################ IndexedOptionArray

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<IndexedOptionArray>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</IndexedOptionArray>\n"
        return out

    def constructor(self):
        return "IndexedOptionArray(" + repr(self.index) + ", " + self.content.constructor() + ")"

################################################################ RecordArray

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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<RecordArray>\n"
        if len(self.contents) == 0:
            out += indent + "    <length>" + str(self.length) + "</length>\n"
        if self.recordlookup is None:
            for i, content in enumerate(self.contents):
                out += content.tostring_part(indent + "    ", "<content i=\"" + str(i) + "\">", "</content>\n")
        else:
            for i, (key, content) in enumerate(zip(self.recordlookup, self.contents)):
                out += content.tostring_part(indent + "    ", "<content i=\"" + str(i) + "\" key=\"" + repr(key) + "\">", "</content>\n")
        out += indent + "</RecordArray>" + post
        return out

    def constructor(self):
        return "RecordArray([" + ", ".join(x.constructor() for x in self.contents) + "], " + repr(self.recordlookup) + ", " + repr(self.length) + ")"

################################################################ UnionArray

class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, list)
        assert isinstance(index, list)
        assert isinstance(contents, list)
        assert len(index) >= len(tags)   # usually ==
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
        else:
            raise AssertionError(where)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<UnionArray>\n"
        out += indent + "    <tags>" + " ".join(str(x) for x in self.tags) + "</tags>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        for i, content in enumerate(self.contents):
            out += content.tostring_part(indent + "    ", "<content i=\"" + str(i) + "\">", "</content>\n")
        out += indent + "</UnionArray>" + post
        return out

    def constructor(self):
        return "UnionArray(" + repr(self.tags) + ", " + repr(self.index) + ", [" + ", ".join(x.constructor() for x in self.contents) + "])"

################################################################ SlicedArray
# (Does not exist, but part of the Uproot Milestone.)

################################################################ ChunkedArray
# (Does not exist, but part of the Uproot Milestone.)

################################################################ PyVirtualArray
# (Does not exist, but part of the Uproot Milestone.)

################################################################ UnmaskedArray
# (Does not exist, but part of the Arrow Milestone.)

################################################################ ByteMaskedArray
# (Does not exist, but part of the Arrow Milestone.)

################################################################ BitMaskedArray
# (Does not exist, but part of the Arrow Milestone.)

################################################################ RedirectArray
# (Does not exist.)

################################################################ SparseUnionArray
# (Does not exist.)

################################################################ SparseArray
# (Does not exist.)

################################################################ RegularChunkedArray
# (Does not exist.)

################################################################ AmorphousChunkedArray
# (Does not exist.)

################################################################ test iter

# for x in range(100):
#     q = list(Content.random())

################################################################ test count

def count(data, axis=0):
    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")

    if isinstance(data, (list, Content)):
        if axis == 0:
            if all(isinstance(x, list) for x in data):
                return [len(x) for x in data]
            else:
                raise IndexError("axis > list depth of structure")
        else:
            return [count(x, axis - 1) for x in data]

    elif isinstance(data, tuple):
        return tuple(count(x, axis) for x in data)

    elif isinstance(data, dict):
        return {n: count(x, axis) for n, x in data.items()}

    elif isinstance(data, (bool, int, float)):
        raise IndexError("axis > list depth of structure")

    else:
        raise NotImplementedError(repr(data))

def RawArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        raise IndexError("axis > list depth of structure")

RawArray.count = RawArray_count

def NumpyArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        if len(self.shape) == 1:
            raise IndexError("axis > list depth of structure")
        if axis == 0:
            return NumpyArray.onedim([self.shape[1]] * self.shape[0])
        else:
            if self.shape[0] == 0:
                return NumpyArray.onedim([])
            else:
                ### Awkward solution
                # content = NumpyArray(self.ptr, self.shape[1:], self.strides[1:], self.offset).count(axis - 1)
                # index = [0] * self.shape[0] * self.shape[1]
                # return RegularArray(IndexedArray(index, content), self.shape[1])

                ### pure NumPy solution
                next = NumpyArray(self.ptr, self.shape[1:], self.strides[1:], self.offset).count(axis - 1)
                shape = [self.shape[0]] + next.shape
                strides = [0] + next.strides   # a good use for stride == 0
                return NumpyArray(next.ptr, shape, strides, next.offset)

NumpyArray.count = NumpyArray_count

def EmptyArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        return EmptyArray()

EmptyArray.count = EmptyArray_count

def RegularArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        return NumpyArray.onedim([self.size] * len(self))
    else:
        return RegularArray(self.content.count(axis - 1), self.size)

RegularArray.count = RegularArray_count

def ListArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        out = [self.stops[i] - self.starts[i] for i in range(len(self.starts))]
        return NumpyArray.onedim(out)
    else:
        return ListArray(self.starts, self.stops, self.content.count(axis - 1))

ListArray.count = ListArray_count

def ListOffsetArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        out = [self.offsets[i + 1] - self.offsets[i] for i in range(len(self.offsets) - 1)]
        return NumpyArray.onedim(out)
    else:
        return ListOffsetArray(self.offsets, self.content.count(axis - 1))

ListOffsetArray.count = ListOffsetArray_count

def IndexedArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        return IndexedArray(self.index, self.content.count(axis))

IndexedArray.count = IndexedArray_count

def IndexedOptionArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        return IndexedOptionArray(self.index, self.content.count(axis))

IndexedOptionArray.count = IndexedOptionArray_count

def RecordArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        contents = []
        for x in self.contents:
            contents.append(x.count(axis))
        return RecordArray(contents, self.recordlookup, self.length)

RecordArray.count = RecordArray_count

def UnionArray_count(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        contents = []
        for x in self.contents:
            contents.append(x.count(axis))
        return UnionArray(self.tags, self.index, contents)

UnionArray.count = UnionArray_count

#for i in range(10):
#    print("count i =", i)
#    array = Content.random()
#    for axis in range(5):
#        print("axis =", axis)
#        try:
#            rowwise = count(array, axis)
#            columnar = array.count(axis)
#        except IndexError:
#            break
#        columnar = list(columnar)
#        assert rowwise == columnar

################################################################ test flatten
def flatten(data, axis=0):
    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")
    if isinstance(data, (list, Content)):
        if axis == 0:
            if all(isinstance(x, list) for x in data):
                return sum(data, [])
            else:
                raise IndexError("cannot concatenate non-lists")
        else:
            return [flatten(x, axis - 1) for x in data]

    elif isinstance(data, tuple):
        return tuple(flatten(x, axis) for x in data)

    elif isinstance(data, dict):
        return {n: flatten(x, axis) for n, x in data.items()}   # does not reduce axis!

    elif isinstance(data, (bool, int, float)):
        raise IndexError("axis > list depth of structure")

    else:
        raise NotImplementedError(repr(data))

    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")

# RawArray
def RawArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        raise IndexError("axis > list depth of structure")

RawArray.flatten = RawArray_flatten

def count_strides(array):
    strides = []
    def recurse(array):
        if isinstance(array, Content):
            strides.append(array.__len__())
            return count_strides(array.__getitem__(0))
        else:
            return array

    recurse(array)

    return strides


def compact_array(array, depth=-1):
    data_items = []

    def recurse(array, depth):
        if isinstance(array, Content) and array.__len__() > 0:
            if depth != 0:
                for it in range(array.__len__()):
                    recurse(array.__getitem__(it), depth - 1)
        else:
            data_items.append(array)

    recurse(array, depth)

    return data_items

def compact_strides(array, shape):
    strides = len(shape)*[1]
    for i in range(len(shape), 1, -1):
        strides[i - 1] = strides[i] * shape[i]
    return strides

def index_array(array, depth=-1):
    indices = []
    offset = array.offset

    def recurse(array, index, depth):
        if isinstance(array, Content) and array.__len__() > 0:
            if depth != 0:
                for it in range(array.__len__()):
                    index = array.offset + it
                    recurse(array.__getitem__(it), index, depth - 1)
        else:
            indices.append(index)

    recurse(array, offset, depth)

    return indices

def partition(tuple, begin, end):
    pivot_idx = begin
    for i in range(begin + 1, end + 1):
        if tuple[i][0] <= tuple[begin][0]:
            pivot_idx += 1
            tuple[i], tuple[pivot_idx] = tuple[pivot_idx], tuple[i]
    tuple[pivot_idx], tuple[begin] = tuple[begin], tuple[pivot_idx]
    return pivot_idx

def quick_sort_recursion(tuple, begin, end):
    if begin >= end:
        return
    pivot_idx = partition(tuple, begin, end)
    quick_sort_recursion(tuple, begin, pivot_idx - 1)
    quick_sort_recursion(tuple, pivot_idx + 1, end)

def quick_sort(tuple, begin = 0, end = None):
    if end is None:
        end = len(tuple) - 1

    return quick_sort_recursion(tuple, begin, end)

# Returns sorted indices
def NumpyArray_argsort(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        tuple = []
        indices = index_array(self, len(self.shape))

        for i in range(len(indices)): {
            tuple.append((self.ptr[indices[i]], indices[i]))
        }

        sorted_tuple = quick_sort(tuple)
        sorted_indices = []
        for it in range(len(tuple)):
            sorted_indices.append(tuple[it][1])

        return NumpyArray.onedim(sorted_indices)

NumpyArray.argsort = NumpyArray_argsort

# NumpyArray
def NumpyArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        if self.shape[0] == 0:
            return NumpyArray.onedim([])

        # compact array and strides
        comp_ptr = compact_array(array, len(self.shape))
        comp_strides = compact_strides(array, self.shape)

        # flatten shape
        shape = []
        if len(self.shape) == 1:
            shape = []
        else:
            offset = axis
            indx = offset
            for i in range(axis):
                shape.append(self.shape[i])
            shape.append(self.shape[axis]*self.shape[axis + 1])
            for i in range(len(self.shape) - axis - 2):
                shape.append(self.shape[axis + 2 + i])

        # flatten strides
        strides = []
        if len(self.strides) == 1:
            strides = []
        else:
            for i in range(axis):
                strides.append(comp_strides[i])
            for i in range(len(comp_strides) - axis - 1):
                strides.append(comp_strides[axis + i + 1])

        return NumpyArray(comp_ptr, shape, strides, 0)

NumpyArray.flatten = NumpyArray_flatten

# EmptyArray
def EmptyArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        return EmptyArray()

EmptyArray.flatten = EmptyArray_flatten

# RegularArray
def RegularArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        if self.content.__len__() % self.size != 0:
            return self.content[0:self.__len__()*self.size]
        else:
            return self.content
    else:
        count = [self.size]*len(self.content)
        ccount = self.count(0)
        offsets = [0]*(self.content.__len__() + 1)
        for i in range(self.content.__len__()):
            l = 0
            for j in range(count[i]):
                l += ccount[j + i*self.size]
                offsets[i + 1] = l

        return ListOffsetArray(offsets, self.content.flatten(axis - 1))

RegularArray.flatten = RegularArray_flatten

# ListArray
def ListArray_flatten(self, axis=0):
    if len(self.stops) > len(self.starts):
        raise IndexError("cannot flatten starts != stops")
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        indxarray = [ x + z for x, y in zip(self.starts, self.stops) for z in range(y - x) if x >= 0 and y >= 0 and y - x > 0]

        return IndexedArray(indxarray, self.content)
    else:
        return ListArray(self.starts, self.stops, self.content.flatten(axis - 1))

ListArray.flatten = ListArray_flatten

# ListOffsetArray
def ListOffsetArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        if self.__len__() > 0:
            starts, stops = zip(*[(self.offsets[i], self.offsets[i] + self.__getitem__(i).__len__()) for i in range(self.__len__()) if self.__getitem__(i).__len__() > 0])

            return ListArray(list(starts), list(stops), self.content).flatten()
        else:
            return self
    else:
      return self.content.flatten(axis - 1)

ListOffsetArray.flatten = ListOffsetArray_flatten

# IndexedArray
def IndexedArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    if axis == 0:
        nextcarry = [x+y for x in self.index for y in range(self.content.__getitem__(x).__len__()) if self.index[x] >= 0 and self.index[x] < len(self.index)]

        return IndexedArray(nextcarry, self.content.flatten(axis))
    else:
        return self.content.flatten(axis - 1)

IndexedArray.flatten = IndexedArray_flatten

# IndexedOptionArray
def IndexedOptionArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    if axis == 0:
        nextcarry = [x for x in self.index if self.index[x] > 0 and self.index[x] < len(self.index)]

        return IndexedOptionArray(nextcarry, self.content)
    else:
        return self.content.flatten(axis - 1)

IndexedOptionArray.flatten = IndexedOptionArray_flatten

# RecordArray
def RecordArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        contents = []
        tocontent = []
        for x in self.contents:
            tocontent = x.flatten(axis)
            if len(x) != len(tocontent):
                return RecordArray(self.contents, self.recordlookup, self.length)
            contents.append(tocontent)

        return RecordArray(contents, self.recordlookup, self.length)

RecordArray.flatten = RecordArray_flatten

# UnionArray
def UnionArray_flatten(self, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        tocontents = [ x.flatten(axis) for x in self.contents if x.__len__() != x.flatten(axis).__len__()]
        toindex = [ tocontents[x].__len__() for x in self.index]

        return UnionArray(self.tags, toindex, tocontents)

UnionArray.flatten = UnionArray_flatten

for i in range(100):
    print("flatten i =", i)
    array = Content.random()
    # OK: array = RegularArray(IndexedOptionArray([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], NumpyArray([2.1, 8.4, 7.4, 1.6, 2.2, 3.4, 6.2, 5.4, 1.5, 3.9, 3.8, 3.0, 8.5, 6.9, 4.3, 3.6, 6.7, 1.8, 3.2], [18], [1], 1)), 3)
    # OK: array = ListOffsetArray([4, 4, 15, 24, 28, 28, 56, 57, 78], RecordArray([], None, 106))
    # OK: array = IndexedArray([4, 4, 4, 6, 0, 6, 4, 6, 2, 4, 5], ListOffsetArray([4, 4, 15, 24, 28, 28, 56, 57, 78], RecordArray([], None, 106)))
    # OK: array = IndexedArray([2, 2, 1, 0, 2], ListArray([5, 2, 10], [11, 6, 11], ListArray([10, 3, 0, 1, 5, 1, 8, 0, 2, 6, 3], [10, 3, 0, 1, 5, 1, 8, 0, 2, 6, 3], EmptyArray())))
    # OK: array = RecordArray([ListOffsetArray([5, 6, 13], RegularArray(RegularArray(UnionArray([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0], [14, 16, 0, 1, 8, 2, 5, 0, 0, 11, 15, 1, 1, 4, 5, 2, 13, 11, 0, 0, 0, 2, 0, 0, 2, 12, 0, 14, 12, 4, 16, 3, 0, 4, 1, 4, 2, 3, 13, 1, 1, 1, 3, 3, 4, 0, 7, 0, 3, 3, 1, 3, 4, 9, 0, 13, 0, 9, 10, 3, 2, 7, 4, 3, 0, 4, 1, 15, 12, 16, 1, 0, 8, 8, 10, 0, 0, 1, 14, 14, 1, 2, 4, 10, 4, 10, 13, 0, 2, 1, 8, 6, 2, 4, 2, 15, 15, 4, 3, 2, 4, 12, 2, 2, 3, 16, 11, 16, 10, 6, 16, 1, 3, 4, 1, 1, 4, 0, 2, 14, 4, 10, 1, 12, 11, 1, 0, 0, 2, 1, 3, 1, 3, 1, 2, 8, 2, 3, 16, 4, 0, 9, 1, 4, 1, 3, 1, 9, 2, 3, 3, 2, 1, 2, 16, 5, 4, 7, 5, 1, 3, 3, 0, 6, 4, 14, 13, 2, 15, 5, 14, 3, 13, 3, 1, 2, 2, 3, 3, 1, 13, 3, 1, 3, 2, 7, 2, 15, 4, 12, 16, 1, 16, 10, 2, 3, 0, 0, 3, 16, 2, 1, 13, 0, 1, 1, 7, 0, 4, 2, 5, 3, 2, 14, 0, 3, 6, 2, 0, 2, 13, 2, 3, 0, 2, 3, 8, 2, 16, 5, 3, 1, 4, 11, 13, 11, 4, 3, 13, 0, 14, 7, 3, 3, 10, 4, 9, 1, 8, 12, 2, 4, 10, 14, 2, 15, 3, 4, 11, 2, 8, 2, 0, 6, 3, 9, 6, 2, 12, 4, 1, 1, 1, 1, 3, 6, 4, 2, 16, 3, 12, 2, 1, 15, 3, 12, 7, 3, 1, 9, 1, 3, 4, 3, 2, 8, 4, 1, 4, 0, 2, 3, 2, 5, 2, 2, 7, 2, 3, 1, 9, 15, 0, 4, 5, 1, 6, 3, 13, 5, 15, 4, 1], [ListArray([4, 2, 6, 7, 4], [12, 2, 12, 12, 4], ListArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], IndexedArray([13], IndexedArray([12, 16, 16, 7, 12, 9, 17, 4, 16, 7, 4, 8, 0, 11, 12, 17], UnionArray([1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 6, 0, 1, 5, 4, 8, 0, 0, 4, 0, 0, 0, 0], [RawArray([5.1, 7.3, 4.5, 7.1, 0.8, 3.8, 5.3, 7.7, 2.9, 7.8]), IndexedOptionArray([0], ListArray([2, 3, 2, 1, 2, 2, 4, 3, 1, 2, 1, 3, 0, 0, 4, 0], [5, 5, 5, 5, 3, 5, 5, 5, 4, 5, 2, 5, 5, 5, 5, 3], IndexedArray([2, 3, 3, 0, 0], IndexedOptionArray([8, 3, 1, 9, 13, 8], RecordArray([], None, 16)))))]))))), ListOffsetArray([0, 3, 20, 21, 21, 24, 27, 31, 33, 46, 54, 60, 60, 77, 77, 77, 81, 92], ListArray([0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1], [2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 2, 1, 2, 0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2], NumpyArray([5.3, 2.3, 2.9, 2.8, 5.0, 2.3, 4.2, 11.1, 8.5, 4.7, 6.3, 2.2, 4.6, 9.0, 2.8, 3.2, 5.6], [2, 2], [2, 1], 13)))]), 3), 2)), EmptyArray()], None, None)
    # OK: array = UnionArray([1, 1, 1], [18, 7, 7], [RegularArray(UnionArray([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [23, 35, 25, 3, 1, 31, 32, 2, 20, 6, 27, 5, 2, 31, 12, 5, 1, 16, 0, 0, 4, 21, 5, 10, 1, 24, 1, 5, 16, 35, 1, 4, 27, 15, 29, 0, 3, 15, 2, 3, 2, 0, 1, 1, 4, 0, 1, 10, 14, 23, 32, 36, 2, 23], [RecordArray([], None, 37), RawArray([7.1, 3.5, 5.0, 4.1, 7.6, 11.4])]), 2), ListArray([3, 2, 2, 0, 2, 1, 3, 3, 1, 2, 1, 2, 1, 2, 3, 0, 1, 1, 3], [4, 4, 3, 1, 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], IndexedOptionArray([6, -4, 1, -13], IndexedOptionArray([0, 0, 2, 0, -16, 0, 2, -50], NumpyArray([3.3, 11.4, 9.5, 1.9, 8.6, 1.8, 3.4, 10.2, 5.6, 3.9, 3.8, 2.6, 5.5, -2.3, 4.1, 10.9], [4, 1, 1], [3, 1, 1], 4))))])
    # OK: array = UnionArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ListArray([], [], ListOffsetArray([7, 12, 27, 70, 70, 90, 93, 94, 95], IndexedOptionArray([-8, 2, 3, 0, 3, -17, -18, 2, 1, 1, -5, 0, -20, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 3, -2, 3, -2, 2, 1, -1, 1, 1, 3, 3, -38, 1, -20, 2, 3, -16, 2, -3, 1, 3, 3, 3, 1, 3, 0, 1, 2, 2, -3, 2, -32, 1, 2, -7, 0, 0, 3, 2, 3, 2, -6, 1, 0, -1, 1, -6, 1, -4, -18, -4, 1, -4, 3, 3, 0, 1, 3, -22, 2, 2, 1, 3, 2, 3, 0, -9, 3, 3, 0, 3, 0, -40, 0], ListOffsetArray([4, 7, 8, 12, 12], RecordArray([IndexedArray([7, 0, 6, 1, 4, 5, 2, 7, 2, 0, 2, 8, 1, 6, 1, 0, 8, 0, 0, 4, 6, 8, 1, 7, 3, 4, 4, 7, 3, 2, 4, 8, 5, 5, 6], IndexedArray([3, 5, 5, 5, 3, 5, 6, 1, 5], NumpyArray([3.4, 11.0, 3.0, 1.4, 7.3, 5.4, 5.7, 6.8, 4.9, 5.3, 6.4, 7.7, 7.9, 5.0, 5.1, 2.0, 10.8, 7.7, 7.6, 6.8, 5.3, 8.7, 7.9, 4.9, 7.3, 5.1, 1.8, 3.0, 7.0, 7.7, 0.7, 0.9, 7.2, 3.1, 4.6, 1.8, 2.6, 2.1, 7.6, 4.6, 6.0], [7, 3], [3, 1], 20)))], None, None))))), ListOffsetArray([37, 44], ListOffsetArray([1, 2, 12, 20, 27, 33, 50, 63, 71, 109, 114, 133, 136, 143, 150, 155, 159, 162, 165, 171, 180, 190, 199, 204, 236, 240, 240, 255, 255, 255, 263, 266, 274, 282, 287, 297, 297, 305, 308, 320, 321, 323, 331, 345, 356, 360, 363, 374, 384], UnionArray([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ListOffsetArray([6, 36], RawArray([5.5, 6.6, 6.1, 7.5, 6.4, 0.2, 3.2, 6.5, 5.7, 2.1, 5.3, 3.5, 6.5, 5.7, 4.9, 3.7, 1.8, 8.2, 7.0, 3.1, 5.3, 3.5, 10.0, 5.7, 6.0, 0.7, 6.1, 9.0, -2.0, 6.3, 5.4, 5.1, 6.8, 3.6, 8.5, 4.4, 11.4, 8.9, 2.9, 5.3, 5.7, 6.9, 5.8, 4.5, 5.0, -0.5, 4.7, 6.9, 8.0, 3.5, 4.6])), IndexedArray([0], RegularArray(NumpyArray([2.9, 2.9, 0.6, 5.7, 10.5, 4.0, 7.0, -0.4, 2.7, 7.4, -1.5, 6.3, 6.8, 0.8, 2.8, 5.1, 7.1, 3.3, 2.3, 0.7, 7.2, 4.8, 6.0, 3.8, 8.9, 11.3, 4.2, 5.9, 3.3, 3.1, 4.5, 3.1, 5.1, 7.6, 2.7, 13.5, 6.0, 5.4, 0.9, 9.2, 9.3, 2.6, 10.4, 5.2, 3.3, 3.0, 4.6, 3.2, 4.5, 6.7, 8.2, 2.5, 3.4, 3.8, 8.9, 3.6, 4.7, 6.6, 6.7, 5.7, 8.8, 13.3, 2.5, 7.5, -2.7, 9.2, 4.4, 0.8, 5.8, 6.6, 5.8, 4.6, 7.8, 8.1, 5.8, 11.6, 6.2, 5.5, 10.0], [13, 1, 3], [6, 5, 1], 1), 2))]))), ListArray([8, 4, 2, 10, 2, 0, 9, 0, 5, 8, 7, 4, 2, 1, 6, 3, 5, 0, 4, 9, 2, 1, 1, 5, 5, 3, 10], [8, 4, 2, 10, 2, 0, 9, 0, 5, 8, 7, 4, 2, 1, 6, 3, 5, 0, 4, 9, 2, 1, 1, 5, 5, 3, 10], EmptyArray())])

    for axis in range(1):
        print("axis =", axis)
        try:
            rowwise = flatten(array, axis)
            columnar = array.flatten(axis)
            #indexed = array.argsort(axis)
        except IndexError:
            break
        columnar = list(columnar)
        assert rowwise == columnar

# ### Don't worry about the not-implemented-yet ones
# # SlicedArray
# # ChunkedArray
# # PyVirtualArray
# # UnmaskedArray
# # ByteMaskedArray
# # BitMaskedArray
# # RedirectArray
# # SparseUnionArray
# # SparseArray
# # RegularChunkedArray
# # AmorphousChunkedArray
