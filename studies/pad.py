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

################################################################ test pad
def pad(data, length, axis=0):
    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")
    else:
        if isinstance(data, (list, Content)):
            if axis == 0:
                result = [x for x in data]
                return result + [None] * length
            else:
                return [pad(x, length, axis - 1) for x in data]
        elif isinstance(data, tuple):
            return tuple(pad(x, length, axis) for x in data)

        elif isinstance(data, dict):
            return {n: pad(x, length, axis) for n, x in data.items()}
        elif isinstance(data, (bool, int, float)):
            raise IndexError

        raise NotImplementedError(repr(data))

# RawArray
def RawArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        raise IndexError("axis > list depth of structure")

RawArray.pad = RawArray_pad

def numpy_array_index(array):
    bytepos = []
    for i in range(array.shape[0]) :
        bytepos.append(i*array.strides[0])

    index = []
    for x in range(len(bytepos)):
        for y in range(array.strides[0]):
            index.append(x*array.strides[0] + array.offset + y)

    return index

def wrap_array(array, stride):
    offsets = []
    for x in range(int(array.__len__()/stride)):
        offsets.append(x*stride)
    offsets.append(array.__len__())

    return ListOffsetArray(offsets, array)

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

# NumpyArray
def NumpyArray_pad(self, length, axis=0):
    # use shapeless padding
    return self.pad_no_shape(length, axis)

    # this is for padding with a NumpyArray shape
    if axis < 0:
        raise NotImplementedError
    elif axis > len(self.shape) - 1:
        raise IndexError
    else:
        # compact array
        comp_ptr = compact_array(self, len(self.shape))
        out = RawArray(comp_ptr)
        comp_index = [x for x in range(out.__len__())]

        # calculate extra padding
        extra_padding = 1
        for x in self.shape[axis+1:]:
            extra_padding = extra_padding * x

        # and how many padding chunks
        chunks = 1
        for x in self.shape[:axis]:
            chunks = chunks * x

        # insert the None's
        insert_index = self.offset
        for x in range(chunks):
            insert_index = insert_index + self.shape[axis] * self.strides[axis]
            for y in range(length):
                for z in range(extra_padding):
                    comp_index.insert(insert_index, -1)
                    insert_index = insert_index + 1

        indexedArray = IndexedOptionArray(comp_index, out)

        # if the array is one-dimentional we are done
        if len(self.shape) == 1:
            return indexedArray

        # else wrap it in a new shape
        else:
            out = indexedArray

            shape = [x for x in self.shape]
            shape[axis] = self.shape[axis] + length

            for x in shape[-1:0:-1]:
                out = wrap_array(out, x)

            return wrap_array(out, shape[0]).content

NumpyArray.pad = NumpyArray_pad

def NumpyArray_pad_no_shape(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis > len(self.shape) - 1:
        raise IndexError
    else:
        # compact array
        comp_ptr = compact_array(self, len(self.shape))
        out = RawArray(comp_ptr)

        # shape it in a RegularArray
        for i in range(len(self.shape) - 1):
            out = RegularArray(out, self.shape[len(self.shape) - 1 - i])

        # index it with shapeless padding
        indxarray = []
        for i in range(out.__len__()):
            indxarray.append(i)

    if axis == 0:

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, out)
    else:
        comp_index = [x for x in range(RawArray(comp_ptr).__len__())]

        # how many padding chunks
        chunks = 1
        for x in self.shape[:axis]:
            chunks = chunks * x

        # insert the None's
        insert_index = self.offset
        for x in range(chunks):
            insert_index = insert_index + self.shape[axis] * self.strides[axis]
            for y in range(length):
                comp_index.insert(insert_index, -1)
                insert_index = insert_index + 1

        return IndexedOptionArray(comp_index, RawArray(comp_ptr))

NumpyArray.pad_no_shape = NumpyArray_pad_no_shape

# EmptyArray
def EmptyArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    else:
        indxarray = []
        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self)

EmptyArray.pad = EmptyArray_pad

# RegularArray
def RegularArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        indxarray = []
        for x in range(self.__len__()):
            indxarray.append(x)
        for y in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self)

    else:
        indxarray = []
        for x in range(self.__len__()):
            indxarray.append(x)

        return IndexedOptionArray(indxarray, self.content.pad(length, axis))

RegularArray.pad = RegularArray_pad

# ListArray
def ListArray_pad(self, length, axis=0):
    if len(self.stops) > len(self.starts):
        raise IndexError("cannot pad starts != stops")
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        indxarray = []
        for x in range(self.starts.__len__()):
            indxarray.append(x)

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self)
    else:
        out = self.content.pad(length, axis-1)
        padded_starts.append(out.__len__())
        padded_stops.append(out.__len__() + length)
        return ListArray(padded_starts, padded_stops, out)

ListArray.pad = ListArray_pad

# ListOffsetArray
def ListOffsetArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        indxarray = []
        for x in range(self.offsets.__len__() - 1):
            indxarray.append(x)

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self)
    else:
        #FIXME: check if self.__len__() > 1:
        starts, stops = zip(*[(self.offsets[i], self.offsets[i] + self.__getitem__(i).__len__()) for i in range(self.__len__())])
        padded_starts = list(starts)
        padded_starts.append(self.content.__len__())
        padded_stops = list(stops)
        padded_stops.append(self.content.__len__() + length)

        return ListArray(padded_starts, padded_stops, self.content.pad(length, axis-1))

ListOffsetArray.pad = ListOffsetArray_pad

# IndexedArray
def IndexedArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    if axis == 0:
        indxarray = []
        for x in self.index:
            indxarray.append(x)

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self.content)
    else:
        out = self.content.pad(length, axis)

        return IndexedArray(self.index, out)

IndexedArray.pad = IndexedArray_pad

# IndexedOptionArray
def IndexedOptionArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    if axis == 0:
        indxarray = []
        for x in self.index:
            indxarray.append(x)

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self.content)
    else:
        return self.content.pad(length, axis-1)

IndexedOptionArray.pad = IndexedOptionArray_pad

# RecordArray
def RecordArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:

        indxarray = []
        for x in range(self.__len__()):
            indxarray.append(x)

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self)
    else:
        raise NotImplementedError

RecordArray.pad = RecordArray_pad

# UnionArray
def UnionArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:

        indxarray = []
        for x in range(self.__len__()):
            indxarray.append(x)

        for i in range(length):
            indxarray.append(-1)

        return IndexedOptionArray(indxarray, self)
    else:
        raise NotImplementedError

UnionArray.pad = UnionArray_pad

for i in range(100):
    print("pad i =", i)
    array = Content.random()
    for axis in range(5):
        axis = 0
        print("axis =", axis)
        try:
            rowwise = pad(array, 3, axis)
            columnar = array.pad(3, axis)
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
