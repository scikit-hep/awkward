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
                if len(data) < length:
                    result = [x for x in data]
                    return result + [None] * (length - len(data))
                else:
                    result = []
                    for x in data:
                        if len(result) < length:
                            result.append(x)
                    return result
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
    # use clipped padding
    return self.pad_and_clip(length, axis)

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
        insert_index = 0
        for x in range(chunks):
            insert_index = insert_index + self.shape[axis] * self.strides[axis]
            for y in range(length):
                comp_index.insert(insert_index, -1)
                insert_index = insert_index + 1

        out = IndexedOptionArray(comp_index, RawArray(comp_ptr))

        shape = [x for x in self.shape]
        shape[axis] = self.shape[axis] + length

        for x in shape[-1:0:-1]:
            out = wrap_array(out, x)

        return out

NumpyArray.pad_no_shape = NumpyArray_pad_no_shape

def NumpyArray_pad_and_clip(self, length, axis=0):
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
        if length > out.__len__():
            for i in range(length - out.__len__()):
                indxarray.append(-1)
        else:
            indxarray = indxarray[:length]

        return IndexedOptionArray(indxarray, out)
    else:
        comp_index = [x for x in range(RawArray(comp_ptr).__len__())]

        # how many padding chunks
        chunks = 1
        for x in self.shape[:axis]:
            chunks = chunks * x

        if length > self.shape[axis]:
            # insert the None's
            insert_index = 0
            for x in range(chunks):
                insert_index = insert_index + self.shape[axis] * self.strides[axis]
                for y in range(length - self.shape[axis]):
                    comp_index.insert(insert_index, -1)
                    insert_index = insert_index + 1
        else:
            # clip
            el = self.shape[axis]
            for i in range(self.shape[axis] - length):
                del comp_index[::el]
                el = el - 1

        out = IndexedOptionArray(comp_index, RawArray(comp_ptr))

        shape = [x for x in self.shape]
        shape[axis] = length

        for x in shape[-1:0:-1]:
            out = wrap_array(out, x)

        return out

NumpyArray.pad_and_clip = NumpyArray_pad_and_clip

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

# ListArray functions
def starts_stops_to_index(starts, stops):
    toindex = []
    for x in range(len(starts)):
        if stops[x] - starts[x] > 0:
            for y in range(stops[x] - starts[x]):
                toindex.append(starts[x] + y)
        else:
            toindex.append(starts[x])

    return toindex

def index_append_pad(index, fromlength, tolength):
    toindex = []
    if tolength > fromlength:
        for i in index:
            toindex.append(i)
        for i in range(tolength - fromlength):
            toindex.append(-1)
    else:
        toindex = index

    return toindex

def index_inject_pad(index, length, starts, stops):
    toindex = []
    for i in index:
        toindex.append(i)

    insert_index = 0
    for i in range(len(starts)):
        insert_index = stops[i]
        for j in range(length):
            toindex.insert(insert_index, -1)
            insert_index = insert_index + 1

    return toindex

def index_pad_to_length(index, tolength, starts, stops):
    toindex = []
    shift = 0
    for i in range(len(starts)):
        if stops[i] - starts[i] <= tolength:
            k = 0
            for j in range(stops[i] - starts[i]):
                toindex.append(index[shift])
                shift = shift + 1
                k = k + 1
            while k < tolength:
                toindex.append(-1)
                k = k + 1
        else:
            for j in range(tolength):
                toindex.append(index[shift])
                shift = shift + 1

    return toindex

def append_starts_stops(starts, stops, array, tolength):
    outstarts = []
    outstops = []
    if tolength > array.__len__():
        for x in starts:
            outstarts.append(x)
        for y in stops:
            outstops.append(y)

        for i in range(tolength - array.__len__()):
            outstarts.append(array.content.__len__() + i)
            outstops.append(outstarts[-1] + 1)
    else:
        for i in range(tolength):
            outstarts.append(starts[i])
            outstops.append(stops[i])

    return outstarts, outstops

def inject_starts_stops(starts, stops, length):
    outstarts = []
    outstops = []
    for x in starts:
        outstarts.append(x)
    for y in stops:
        outstops.append(y + length)

    return outstarts, outstops

def pad_starts_stops(starts, stops, length):
    outstarts = []
    outstops = []
    for x in starts:
        outstarts.append(x)
    for y in range(len(stops)):
        outstops.append(starts[y] + length)

    return outstarts, outstops

def new_starts_stops(starts, stops, length):
    outstarts = []
    outstops = []
    for x in range(len(starts)):
        outstarts.append(x*length)
        outstops.append(outstarts[-1] + length)

    return outstarts, outstops

# ListArray
def ListArray_pad(self, length, axis=0):
    if len(self.stops) > len(self.starts):
        raise IndexError("cannot pad starts != stops")
    if axis < 0:
        raise NotImplementedError

    elif axis == 0:
        if length > self.__len__():
            outindex = starts_stops_to_index(self.starts, self.stops)
            outindex = index_append_pad(outindex, self.__len__(), length)
            out = IndexedOptionArray(outindex, self.content)

            starts, stops = append_starts_stops(self.starts, self.stops, self, length)

            return ListArray(starts, stops, out)
        else:
            starts, stops = append_starts_stops(self.starts, self.stops, self, length)
            return ListArray(starts, stops, self.content)

    elif axis == 1:

        outindex = starts_stops_to_index(self.starts, self.stops)
        outindex = index_pad_to_length(outindex, length, self.starts, self.stops)
        out = IndexedOptionArray(outindex, self.content)

        starts, stops = new_starts_stops(self.starts, self.stops, length)

        return ListArray(starts, stops, out)

    else:

        return ListArray(self.starts, self.stops, self.content.pad(length, axis - 1))

ListArray.pad = ListArray_pad

# ListOffsetArray functions
def offsets_to_index(offsets):
    toindex = []
    for x in range(len(offsets)):
        toindex.append(offsets[x])

    return toindex

def offsets_to_starts_stops(offsets):
    starts = []
    stops = []
    for x in range(len(offsets) - 1):
        starts.append(offsets[x])
    for x in range(len(starts)):
        stops.append(offsets[x + 1])

    return starts, stops

# ListOffsetArray
def ListOffsetArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:
        if length > self.__len__():
            starts, stops = offsets_to_starts_stops(self.offsets)

            outindex = []
            if isinstance(self.content, IndexedOptionArray):
                outindex = self.content.index
                for i in range(length - self.__len__()):
                    outindex.append(-1)
                    starts.append(len(outindex)-1)
                    stops.append(len(outindex))

                out = IndexedOptionArray(outindex, self.content.content)

                return ListArray(starts, stops, out)
        else:

            return ListOffsetArray(self.offsets[:length+1], self.content)
    else:
        out = self.content.pad(length, axis)

        #FIXME: check if self.__len__() > 1:
        starts, stops = zip(*[(self.offsets[i], self.offsets[i] + self.__getitem__(i).__len__()) for i in range(self.__len__())])
        padded_starts = list(starts)
        padded_starts.append(self.content.__len__())
        padded_stops = list(stops)
        padded_stops.append(out.__len__())

        return ListArray(padded_starts, padded_stops, out)

ListOffsetArray.pad = ListOffsetArray_pad

# IndexedArray
def IndexedArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    if axis == 0:
        outindex = []
        if length > self.__len__():
            outindex = index_append_pad(self.index, self.__len__(), length)
        else:
            outindex = self.index[:length]

        return IndexedOptionArray(outindex, self.content)

    else:
        return self.content.pad(length, axis - 1)

IndexedArray.pad = IndexedArray_pad

# IndexedOptionArray
def IndexedOptionArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    if axis == 0:
        outindex = []
        if length > self.__len__():
            outindex = index_append_pad(self.index, self.__len__(), length)
        else:
            outindex = self.index[:length]

        return IndexedOptionArray(outindex, self.content)

    else:
        return self.content.pad(length, axis - 1)

IndexedOptionArray.pad = IndexedOptionArray_pad

# RecordArray
def RecordArray_pad(self, length, axis=0):
    if axis < 0:
        raise NotImplementedError
    elif axis == 0:

        outindex = []
        if length > self.__len__():
            for x in range(self.__len__()):
                outindex.append(x)

            for i in range(length - self.__len__()):
                outindex.append(-1)
        else:
            for x in range(length):
                outindex.append(x)

        return IndexedOptionArray(outindex, self)

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

for i in range(10):
    print("pad i =", i)
    array = Content.random()
    #array = ListArray([1, 0, 3, 5, 1, 3, 4, 1, 4, 1], [6, 1, 6, 6, 6, 6, 6, 3, 6, 6], ListOffsetArray([14, 15, 15, 15, 26, 26, 26], IndexedOptionArray([0, -3, 1, -2, 1, 0, 0, -3, -13, 0, 1, 1, 0, 1, 1, 1, 1, -10, 0, -1, 0, 0, 0, 1, -1, 1, 1], RawArray([1.5, 3.3]))))
    #array = ListArray([4, 11, 9], [12, 12, 12], UnionArray([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0], [UnionArray([1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0], [EmptyArray(), ListOffsetArray([17, 18], RecordArray([], [], 22))]), RegularArray(NumpyArray([6.2, 3.6, 5.6, 3.7, 1.6, 8.4, 2.6, 3.5, 3.6, 9.1, 6.2, 5.7, 5.5, 8.9, 3.9, 7.3], [4, 2], [4, 1], 0), 3)]))
    #array = UnionArray([0], [12], [RawArray([-0.3, 6.9, 7.3, 5.1, 4.4, 5.2, 6.4, 5.7, 3.5, 4.8, 6.8, 2.9, 7.4, 5.9, 7.1, 3.2, 9.2, 1.2, 2.0, 1.0, 4.7, 8.4, 1.7, 9.7, -0.2, 5.9, 8.6, 10.0, 1.7]), EmptyArray()])
    #array = RecordArray([], None, 2)
    #array = NumpyArray([2.3, -0.0, 3.2, 4.1, 0.9, 10.7, 0.2, 8.0, 3.7, 8.2, 4.0, 5.4, 4.8, 5.9, 2.4, 4.6, 6.9, 6.6, 2.3, 11.4, 9.1, 8.2, 3.9, 3.2, 0.8, 7.0, 0.2, 10.5, 3.7, 2.0, 3.5, 5.0, 8.9, 6.5, 3.1, 6.3, 5.2, 3.6, 3.4, 6.0, 9.1, 7.1, 6.0, -0.0, 11.3, 5.0, 6.8, 2.0, 3.9, -0.9, 9.4, 6.0], [10, 2], [3, 1], 22)
    #array = IndexedArray([10, 4, 16, 5], ListOffsetArray([4, 5, 18, 40, 66, 66, 76, 129, 139, 140, 161, 164, 187, 232, 234, 251, 257, 271, 311], ListArray([15, 1, 2, 4, 10, 14, 4, 6, 2, 3, 8, 7, 17, 6, 11, 4, 8, 8, 4, 9, 2, 0, 17, 12, 5, 0, 9, 1, 11, 2, 4, 15, 6, 13, 0, 17, 1, 11, 12, 2, 12, 12, 0, 9, 9, 3, 16, 1, 16, 11, 16, 8, 14, 0, 2, 11, 0, 4, 14, 5, 6, 2, 13, 4, 12, 7, 7, 16, 4, 9, 7, 15, 12, 12, 13, 11, 4, 14, 17, 6, 5, 0, 5, 3, 7, 2, 13, 2, 8, 15, 7, 9, 12, 14, 15, 2, 3, 12, 16, 6, 8, 11, 10, 13, 3, 3, 3, 4, 7, 5, 14, 5, 4, 13, 2, 14, 0, 4, 8, 8, 7, 3, 6, 15, 1, 10, 15, 9, 12, 3, 10, 12, 7, 12, 13, 0, 10, 0, 2, 16, 1, 14, 16, 17, 14, 0, 8, 14, 14, 1, 4, 8, 9, 6, 2, 16, 16, 12, 15, 2, 1, 5, 9, 3, 12, 11, 13, 1, 11, 10, 14, 9, 11, 13, 6, 17, 11, 1, 2, 13, 15, 0, 5, 15, 13, 8, 16, 15, 9, 12, 7, 10, 11, 5, 9, 8, 5, 4, 7, 4, 10, 9, 14, 5, 2, 17, 12, 1, 4, 6, 8, 8, 8, 11, 6, 11, 13, 3, 8, 10, 14, 4, 1, 9, 2, 3, 17, 13, 17, 2, 12, 7, 6, 0, 11, 16, 9, 7, 12, 15, 0, 0, 4, 12, 12, 7, 17, 7, 9, 6, 14, 9, 17, 11, 13, 14, 16, 3, 4, 5, 7, 13, 0, 2, 4, 7, 12, 17, 6, 7, 12, 12, 2, 10, 17, 0, 6, 8, 12, 7, 13, 5, 12, 6, 5, 17, 9, 5, 17, 6, 9, 2, 9, 4, 3, 7, 1, 15, 10, 2, 10, 0, 1, 11, 5, 4, 1, 0, 13, 15, 2, 7, 15, 8, 16, 1, 0, 2, 3, 17, 14, 9], [16, 6, 15, 15, 18, 18, 18, 18, 2, 18, 18, 12, 18, 9, 18, 15, 18, 16, 8, 9, 2, 6, 18, 18, 9, 18, 17, 4, 11, 8, 18, 18, 6, 18, 16, 18, 18, 15, 13, 18, 18, 18, 6, 18, 18, 16, 18, 10, 18, 16, 18, 12, 18, 6, 2, 18, 4, 4, 18, 18, 11, 18, 18, 18, 18, 15, 8, 18, 18, 11, 18, 18, 14, 13, 16, 13, 18, 18, 18, 9, 13, 1, 18, 6, 17, 2, 16, 18, 17, 18, 14, 18, 18, 18, 17, 5, 10, 12, 18, 13, 18, 16, 18, 18, 5, 4, 11, 7, 18, 18, 18, 18, 12, 18, 7, 18, 6, 15, 18, 14, 11, 4, 17, 17, 2, 10, 18, 11, 18, 4, 18, 12, 10, 16, 15, 7, 16, 2, 18, 18, 3, 18, 18, 18, 18, 5, 18, 18, 14, 6, 18, 18, 10, 12, 3, 18, 18, 18, 18, 10, 9, 6, 11, 4, 18, 12, 16, 7, 18, 14, 18, 13, 16, 18, 13, 18, 18, 13, 18, 18, 18, 2, 6, 18, 18, 14, 18, 18, 16, 18, 18, 18, 15, 11, 11, 18, 18, 7, 17, 8, 18, 18, 16, 5, 7, 18, 18, 11, 4, 18, 10, 18, 17, 18, 16, 18, 18, 11, 18, 11, 18, 18, 11, 18, 5, 5, 18, 13, 18, 8, 13, 15, 8, 4, 12, 16, 18, 18, 18, 18, 1, 12, 10, 15, 16, 13, 18, 13, 12, 18, 18, 16, 17, 18, 18, 18, 18, 5, 10, 18, 18, 15, 5, 3, 18, 18, 17, 18, 12, 17, 18, 16, 6, 18, 18, 16, 7, 15, 18, 13, 13, 18, 18, 7, 5, 18, 14, 9, 18, 14, 18, 11, 10, 16, 18, 8, 1, 18, 16, 3, 10, 5, 1, 15, 7, 18, 11, 1, 15, 18, 18, 13, 17, 16, 18, 18, 4, 3, 8, 18, 18, 13], UnionArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 3, 7, 7, 8, 8, 2, 8, 2, 4, 3, 8, 1, 0, 7, 1, 4], [ListArray([1, 1, 1, 0, 1, 0, 0, 0, 1], [2, 2, 1, 2, 2, 2, 2, 2, 2], IndexedArray([17, 10], UnionArray([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [9, 16, 5, 1, 16, 18, 6, 17, 19, 17, 9, 10, 8, 3, 13, 7, 11, 6, 9, 20, 6, 0, 0, 17], [ListOffsetArray([1, 5, 8, 37, 46, 65, 76, 86, 90, 92, 103, 106, 108, 144, 147, 153, 161, 161], ListArray([10, 13, 10, 11, 11, 8, 2, 12, 16, 16, 3, 8, 7, 2, 9, 9, 7, 10, 13, 9, 13, 3, 1, 5, 10, 12, 8, 8, 10, 2, 9, 8, 13, 0, 9, 7, 15, 14, 14, 15, 6, 7, 4, 13, 0, 0, 3, 11, 12, 15, 15, 1, 13, 12, 2, 0, 9, 9, 3, 14, 10, 16, 13, 6, 10, 11, 14, 14, 5, 12, 13, 14, 0, 15, 9, 9, 7, 16, 0, 15, 3, 8, 4, 0, 6, 9, 13, 13, 5, 2, 7, 5, 15, 5, 5, 11, 13, 16, 14, 2, 3, 8, 14, 9, 0, 12, 0, 1, 10, 8, 8, 0, 0, 11, 9, 9, 13, 9, 15, 9, 1, 8, 1, 16, 9, 15, 4, 12, 14, 13, 10, 9, 15, 15, 2, 2, 10, 14, 9, 8, 0, 3, 7, 8, 5, 3, 3, 3, 2, 14, 15, 14, 8, 5, 2, 11, 9, 11, 5, 3, 13, 4], [14, 17, 17, 17, 17, 13, 17, 17, 17, 17, 4, 17, 8, 16, 17, 10, 17, 17, 17, 10, 17, 13, 12, 17, 10, 17, 12, 9, 11, 15, 17, 17, 17, 17, 17, 11, 17, 17, 17, 17, 7, 17, 10, 17, 14, 17, 7, 17, 17, 17, 17, 3, 14, 17, 17, 17, 17, 17, 11, 17, 17, 17, 15, 10, 16, 17, 17, 17, 17, 17, 15, 14, 4, 17, 13, 10, 17, 16, 10, 17, 11, 11, 17, 5, 17, 12, 17, 15, 17, 17, 17, 17, 17, 6, 5, 12, 17, 17, 14, 2, 6, 17, 17, 10, 17, 13, 10, 3, 12, 10, 8, 6, 12, 17, 13, 11, 17, 9, 17, 13, 17, 9, 4, 16, 10, 17, 17, 17, 16, 14, 10, 16, 17, 16, 17, 5, 11, 17, 16, 17, 17, 17, 14, 17, 6, 17, 10, 14, 12, 17, 17, 17, 13, 17, 2, 17, 12, 17, 6, 17, 17, 17], IndexedArray([1, 1, 1, 1, 1, 0, 2, 2, 2, 1, 0, 1, 1, 2, 0, 1, 1], RecordArray([RecordArray([RawArray([3.3, 6.1, 7.6, 3.1, 6.6, 5.1, 4.0, 5.4, 1.6, 8.1, 4.7, 5.8, 9.8, 0.1, 5.0, 3.3, 9.0, 2.1, 2.8, 8.7, 3.5, 5.5, 4.2, 8.6]), IndexedArray([4, 9, 4, 9, 4, 7, 2, 10, 2, 1, 10, 6, 7, 8, 5], RecordArray([], [], 11))], ['x0', 'x1'], None), NumpyArray([10.1, 2.0, 4.4, 7.8, 8.0, 4.5, 9.0, 3.1, 7.4, 4.5, 7.6, 3.0, 6.5, 1.2, 5.3, 11.1, 3.1, 7.1, -2.5, 4.1, 9.3, 9.1, 4.6, 0.2, 7.2, 7.3, 2.8, 8.2, 1.6, 6.0, 2.1, 9.9, 7.8, 1.8, 4.0, 7.5, 5.5, 8.0, 5.6, -0.7, 0.9, 6.4, 3.4, 1.8, 3.1, 4.8, 0.8, 1.7], [3, 1, 2], [3, 3, 1], 39)], ['x0', 'x1'], None)))), IndexedArray([5, 3, 4, 0, 6, 4, 5, 6, 2, 0, 5, 5, 4, 1, 1, 5, 6, 3, 1, 4, 7, 2, 2, 4, 8, 1], ListOffsetArray([18, 22, 24, 42, 47, 69, 80, 92, 117, 118], UnionArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 4, 6, 6, 3, 5, 0, 11, 12, 11, 1, 6, 9, 12, 6, 12, 7, 8, 10, 6, 8, 1, 3, 5, 1, 0, 6, 3, 12, 5, 2, 7, 4, 10, 10, 10, 8, 5, 3, 3, 2, 1, 1, 6, 12, 1, 5, 9, 2, 0, 5, 8, 0, 3, 8, 3, 1, 3, 11, 4, 12, 2, 8, 10, 11, 7, 12, 2, 11, 1, 1, 8, 7, 2, 6, 5, 3, 4, 12, 10, 11, 9, 8, 8, 7, 9, 9, 8, 0, 1, 2, 12, 5, 11, 2, 5, 2, 10, 4, 8, 8, 4, 12, 11, 11, 11, 12, 8, 4, 6, 0, 2, 10, 7, 11, 3, 8, 8, 5, 0], [IndexedArray([2, 2, 1, 1, 1, 2, 0, 2, 0, 2, 1, 2, 2], RawArray([2.2, 5.0, 8.4]))]))), ListArray([4, 1, 2, 3, 0, 2, 0], [5, 5, 5, 5, 1, 5, 2], IndexedOptionArray([-2, -5, 9, 7, 2], IndexedArray([1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1], IndexedArray([12, 14], ListArray([0, 3, 3, 4, 4, 3, 4, 1, 1, 0, 3, 4, 4, 4, 0, 1, 0, 3, 0, 2, 2], [4, 5, 5, 4, 5, 5, 5, 5, 1, 1, 4, 5, 5, 5, 5, 5, 2, 5, 5, 5, 5], RawArray([9.0, 2.6, 6.1, 7.9, 4.8]))))))])))]))))
    #array = IndexedArray([2, 2, 1, 0, 2], ListArray([5, 2, 10], [11, 6, 11], ListArray([10, 3, 0, 1, 5, 1, 8, 0, 2, 6, 3], [10, 3, 0, 1, 5, 1, 8, 0, 2, 6, 3], EmptyArray())))
    #array = ListArray([5, 2, 10], [11, 6, 11], ListArray([10, 3, 0, 1, 5, 1, 8, 0, 2, 6, 3], [10, 3, 0, 1, 5, 1, 8, 0, 2, 6, 3], EmptyArray()))
    #array = IndexedOptionArray([3, -14, -15, -12, 1, 3, 4, 1, 3, 2, 3, 4, 4, 4, 4, -11, 2, 3, 0, 2, 0, 0, 3, -35, 3, 3, 0, 4, 2, 4, 4, 2, 1, 2, 1, 2, 2, 4, 0, 2, 2, -8], RawArray([5.1, 3.8, 4.8, 3.7, 8.3]))
    #array = ListOffsetArray([14, 15, 15, 15, 26, 26, 26], IndexedOptionArray([0, -3, 1, -2, 1, 0, 0, -3, -13, 0, 1, 1, 0, 1, 1, 1, 1, -10, 0, -1, 0, 0, 0, 1, -1, 1, 1], RawArray([1.5, 3.3])))
    array = IndexedArray([46, 1, 77, 85, 63, 73, 57, 7, 87, 86, 0, 4, 72, 82, 14, 37, 37, 68], RecordArray([RecordArray([RegularArray(UnionArray([0, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 0, 1, 1, 1, 1, 2, 1, 2, 0, 1, 2, 2, 1, 2, 0, 1, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 1, 0, 2, 2, 2, 2, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 2, 0, 2, 1, 2, 1, 2, 2, 2, 1, 2, 0, 1, 1, 2, 2, 2, 0, 0, 1, 2, 2, 0, 2, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0, 1, 1, 2, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 2, 2, 2, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 0, 1, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 1, 0, 2, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 0, 2, 0, 0, 1, 0, 2, 0, 2, 2, 1, 0, 0, 0, 2, 2, 0, 0, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 0, 2, 0, 0, 1, 0, 2, 1, 1, 0, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 0, 1, 1, 1, 0, 2, 0, 0, 2, 2, 2, 1, 0, 0, 0, 2, 1, 1, 1, 0, 2, 1, 1, 1, 2, 0, 1, 2, 2, 0, 2, 2, 1, 2, 0, 2, 1, 2, 0, 2, 2, 0, 2, 0, 1, 0, 0, 1, 0, 2, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 2, 2, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 2, 0, 1, 0, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 2, 0, 2, 1, 1, 0, 2, 2, 0, 1, 2, 0, 1, 1, 1, 1, 1, 2, 0, 2, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 2, 2, 1, 0, 1, 2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 0, 0, 1, 2, 0, 0, 1, 1, 0, 2, 1, 1, 1, 0, 2, 1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 1, 2, 1, 2, 2, 1, 0, 2, 0, 1, 1, 2, 2, 1, 2, 0, 0, 1, 2, 2, 1, 2, 1, 0, 2, 0, 1, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1, 0], [2, 0, 15, 12, 16, 1, 11, 0, 0, 0, 1, 0, 1, 11, 14, 13, 17, 1, 11, 0, 2, 6, 0, 1, 1, 1, 1, 11, 1, 0, 2, 1, 1, 0, 1, 2, 2, 0, 4, 1, 5, 0, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0, 1, 0, 1, 2, 14, 7, 0, 2, 1, 11, 0, 17, 0, 0, 1, 0, 0, 9, 1, 2, 6, 0, 2, 0, 13, 0, 4, 0, 0, 1, 0, 17, 5, 0, 0, 0, 6, 0, 14, 1, 0, 0, 14, 1, 1, 10, 6, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 14, 1, 11, 2, 1, 9, 0, 8, 1, 2, 13, 1, 8, 2, 13, 2, 1, 0, 1, 4, 1, 1, 3, 0, 0, 1, 1, 12, 2, 0, 1, 1, 2, 11, 1, 0, 1, 0, 1, 0, 1, 10, 0, 1, 1, 15, 2, 12, 0, 1, 1, 6, 1, 1, 0, 0, 14, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 14, 1, 1, 0, 0, 0, 1, 7, 1, 1, 14, 15, 2, 1, 2, 1, 14, 0, 0, 1, 0, 0, 11, 1, 2, 1, 1, 1, 2, 0, 0, 2, 1, 1, 15, 1, 9, 1, 0, 14, 1, 0, 6, 2, 1, 1, 1, 12, 1, 0, 10, 8, 1, 10, 1, 1, 2, 0, 3, 0, 2, 2, 0, 0, 2, 3, 9, 1, 1, 0, 2, 0, 1, 0, 7, 1, 0, 1, 1, 15, 2, 2, 2, 1, 2, 8, 1, 0, 2, 14, 1, 1, 2, 0, 1, 13, 0, 1, 0, 14, 1, 0, 1, 1, 2, 0, 1, 16, 0, 2, 5, 1, 1, 8, 1, 2, 0, 1, 0, 2, 2, 1, 2, 6, 2, 2, 1, 0, 2, 2, 1, 9, 1, 1, 1, 9, 7, 12, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 0, 16, 0, 1, 1, 0, 16, 3, 1, 0, 0, 7, 2, 1, 13, 1, 8, 0, 1, 1, 1, 1, 9, 6, 3, 1, 1, 2, 0, 2, 5, 0, 0, 0, 0, 16, 0, 2, 6, 2, 6, 1, 1, 0, 2, 0, 0, 5, 0, 0, 12, 7, 12, 0, 0, 14, 2, 1, 7, 1, 0, 0, 1, 5, 1, 17, 1, 8, 1, 0, 0, 0, 6, 2, 8, 10, 9, 15, 15, 1, 0, 10, 0, 5, 0, 0, 12, 1, 1, 1, 13, 10, 0, 0, 16, 8, 13, 1, 0, 0, 0, 1, 0, 1, 0, 5, 1, 1, 1, 16, 1, 6, 0, 1, 17, 2, 0, 2, 5, 11, 0, 1, 1, 0, 0, 2, 12, 1, 1, 2, 1, 6, 1, 1, 0, 0, 2, 0, 0, 0, 11, 13, 1, 2, 16, 6, 17, 0], [IndexedOptionArray([10, -16, 0], NumpyArray([3.2, -0.0, 2.1, 7.8, 1.5, 13.0, -0.5, 0.8, 3.3, 5.8, 7.0, 2.1, 3.6, 5.4, 4.8, 4.8, 7.8, 4.4, 4.9, 6.6, 4.0, -0.6, 6.8, 8.6, 5.3, 3.9, 3.9, 9.4, 6.8, 5.7, 6.0, 4.7, 4.3, 6.6, 6.9, 3.5, 8.3, 11.1, 4.4, 5.2, 8.5, 12.5, 3.8, -1.0, 7.3, 8.1, 2.6, 8.3, 6.5, 6.6, -1.5, 5.4, 3.9, 2.3, 1.6, 3.8, 6.8, 2.6, 10.9, 3.9, 6.7, 4.3, 7.5, 6.1, 7.8, 3.9, 3.6, 8.1, 2.0, 3.3, 1.0, 9.1, 9.7, 3.2, 4.9, 5.4, -0.2, 0.0, 9.0, 2.6, 3.2, 9.1, 1.0, 4.8, 4.1, 5.4, 2.5, 3.0, 5.7, 8.7, 6.4, 3.7, 2.3, 5.2, 5.4, 7.2, 12.2, 8.4, 3.1, 7.0, 3.6, 4.3, 7.0, 2.7, 2.4, 3.0, 3.8, 2.9, 3.9, 5.4, 5.8, 14.1, 1.2, 7.3, 5.5, -0.7, 2.2, 5.8, 4.7, 10.9, 5.3, 11.3, 4.5, 3.9, 6.3, 6.0, 3.0, -0.0, 7.1, 1.5, 7.5, 4.4, 6.3, -1.4, 7.8, 4.1, 8.4, 7.4, 2.9, 2.4, 1.3, 4.8, 1.0, 5.8], [11, 3, 2], [13, 4, 1], 1)), RecordArray([ListOffsetArray([26, 33, 34, 38, 59, 64, 64, 65, 74, 99, 107, 111, 115, 120, 131, 151, 152, 160, 163, 175], ListArray([17, 0, 2, 8, 6, 3, 10, 16, 3, 2, 16, 5, 11, 11, 5, 1, 3, 5, 6, 8, 11, 19, 9, 19, 7, 1, 7, 15, 11, 14, 10, 0, 10, 4, 0, 3, 6, 13, 2, 11, 3, 16, 12, 10, 5, 9, 15, 8, 16, 12, 3, 16, 18, 8, 9, 18, 5, 16, 4, 6, 10, 16, 2, 19, 15, 8, 13, 5, 8, 14, 7, 15, 17, 18, 0, 3, 16, 4, 0, 7, 11, 17, 19, 18, 18, 4, 2, 19, 13, 14, 10, 16, 16, 7, 5, 7, 14, 9, 3, 18, 17, 12, 14, 14, 19, 14, 7, 18, 0, 2, 3, 14, 2, 9, 8, 5, 7, 5, 8, 6, 15, 12, 14, 16, 8, 4, 5, 1, 11, 10, 6, 19, 6, 8, 19, 14, 4, 11, 6, 11, 10, 7, 19, 1, 1, 11, 3, 10, 19, 11, 9, 19, 13, 8, 19, 14, 11, 11, 6, 11, 6, 7, 18, 7, 1, 8, 12, 15, 11, 15, 8, 9, 14, 14, 16, 16], [20, 3, 8, 18, 20, 13, 20, 20, 8, 17, 20, 7, 11, 20, 9, 5, 13, 6, 11, 20, 14, 20, 20, 19, 16, 9, 15, 20, 18, 19, 11, 1, 13, 9, 9, 13, 11, 20, 8, 20, 20, 16, 15, 11, 7, 11, 20, 17, 17, 14, 13, 20, 19, 13, 13, 20, 5, 20, 11, 18, 13, 20, 20, 20, 20, 11, 16, 10, 20, 16, 15, 20, 20, 19, 12, 5, 20, 20, 20, 15, 20, 19, 20, 20, 20, 5, 20, 20, 20, 20, 17, 20, 20, 20, 20, 10, 18, 20, 3, 20, 20, 20, 18, 20, 19, 20, 20, 19, 11, 9, 13, 17, 19, 20, 9, 14, 17, 5, 20, 7, 18, 20, 15, 20, 14, 5, 20, 5, 13, 20, 20, 20, 10, 16, 20, 20, 6, 13, 20, 12, 11, 20, 19, 14, 20, 18, 7, 19, 20, 16, 16, 20, 20, 14, 20, 14, 13, 16, 9, 20, 20, 17, 20, 14, 18, 10, 13, 20, 20, 16, 20, 20, 20, 20, 20, 20], ListOffsetArray([52, 59, 72, 102, 123, 126, 138, 158, 159, 169, 177, 177, 189, 191, 195, 204, 214, 215, 215, 220, 220], IndexedOptionArray([-3, -19, 12, -7, 7, -3, 5, 13, 15, 9, 0, 5, 5, 15, 1, -4, 5, 5, 3, 6, 4, -7, -19, 7, 14, 8, 8, -6, -8, 14, 6, 9, -14, -9, 8, -5, 3, 2, 13, -3, 10, 2, -3, 11, -18, 12, -42, 15, 14, 8, 6, -25, -13, 13, 15, 12, 2, 9, 12, 10, 15, 12, 11, 8, 5, -2, 4, 9, -5, 4, -4, -13, 8, 12, -31, 14, 10, 13, 13, 4, 7, 5, -1, 15, 1, 5, -10, 4, 1, 9, 1, 14, 7, 14, 13, -5, 4, 8, 5, 15, 14, -17, 14, 13, 4, 5, 4, 3, 7, 10, 3, 4, 3, -5, 6, -16, 1, -17, 1, -5, 8, 15, 13, 12, 11, -14, 0, 2, -1, 7, -4, 2, 7, -2, 13, -28, 5, 10, 6, -4, 8, 8, 12, -1, 2, 5, 5, 12, 13, 11, -7, 12, 11, 0, 3, 14, 7, 2, 0, 2, 3, 6, -4, 10, 15, 2, 2, 6, -14, 0, 3, 8, 14, -2, 2, 13, 1, 7, -4, 2, 9, 11, 4, 8, 13, 8, 9, 0, -22, 8, 7, 12, 1, 1, 7, 14, 3, 0, 7, 1, 14, 14, 3, 8, 2, -1, 14, 12, 4, 1, 5, 3, 12, 7, 10, 0, 9, 14, 2, 8, 15, 7, 4], RegularArray(IndexedOptionArray([1, 7, 6, 5, 4, 1, 2, 4, 2, 2, 0, 1, 6, 7, 6, 4, 3, 5, 3, 1, 7, 6, 5, 6, 2, 1, 6, 3, 4, 6, 5, 7, 2], ListOffsetArray([11, 15, 35, 42, 46, 52, 55, 71, 103], IndexedArray([8, 5, 6, 11, 16, 1, 16, 17, 9, 1, 0, 7, 11, 2, 4, 14, 6, 3, 6, 12, 16, 4, 2, 7, 5, 3, 12, 0, 10, 6, 0, 13, 3, 7, 6, 15, 9, 4, 4, 17, 0, 14, 13, 6, 5, 5, 9, 15, 17, 3, 12, 13, 0, 5, 0, 14, 3, 15, 12, 17, 1, 11, 4, 16, 9, 15, 10, 12, 14, 13, 6, 1, 1, 13, 16, 14, 12, 15, 2, 5, 1, 2, 2, 13, 10, 14, 8, 0, 9, 15, 11, 5, 10, 12, 0, 11, 5, 0, 11, 10, 4, 16, 3, 5, 5], UnionArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [6, 0, 10, 0, 6, 6, 14, 0, 6, 3, 7, 10, 7, 6, 5, 0, 13, 11], [RegularArray(IndexedArray([7, 0, 11, 0, 13, 5, 13, 6, 1, 4, 10, 5, 10, 7, 1, 1], IndexedOptionArray([15, 16, 1, 7, 15, 9, -11, -12, -7, 6, 8, 13, -16, 6], NumpyArray([3.0, 5.1, 3.2, 3.2, 7.9, 8.7, 7.2, 5.6, 5.1, 4.9, 7.9, 1.9, 9.0, 3.1, 4.9, 5.1, 0.5, 5.8, 7.2, 4.9, 4.6, 6.7, 0.7, 7.7, 2.6, 7.4, 4.0, 1.0, 4.4, 3.5, 5.2, 11.9, 4.0], [22], [1], 11))), 1)])))), 2))))), ListOffsetArray([37, 44, 64, 69, 90, 91, 99, 133, 141, 143, 144, 151, 153, 156, 157, 170, 176, 225, 236], RawArray([6.0, 1.7, 7.0, 4.2, 6.5, 4.3, 1.1, 6.9, -5.2, -1.4, 7.3, 5.5, 1.5, 4.1, 5.0, 2.9, 7.1, 3.4, 2.3, 2.3, 0.5, 10.1, 3.9, 5.8, 9.2, 9.8, 4.9, 5.5, 6.7, 4.2, 4.6, 10.4, 3.8, 6.4, 2.3, 0.1, 2.5, 0.6, -5.1, 3.5, 0.4, 2.1, 4.5, 2.3, 9.5, -1.0, 5.8, 4.2, 5.6, 7.6, 3.0, 5.7, 8.5, 7.7, 3.6, 2.0, 0.8, 7.4, 9.5, 0.5, 4.6, 3.7, 3.4, 8.5, 1.1, 7.3, 8.3, 5.2, 9.0, 5.5, 8.3, 6.9, 8.6, 0.7, 9.0, 4.9, 6.4, 3.9, 7.1, 3.3, 3.7, 9.7, 8.3, 7.0, 3.1, 3.2, 5.0, 4.4, 3.0, 4.9, -4.2, 8.3, 4.9, 9.2, 0.4, 3.0, 6.4, 5.3, 4.8, 2.8, 8.1, 11.5, 3.1, 0.8, 5.9, 5.4, 7.8, 9.7, 1.8, 5.8, 2.2, -0.6, 1.3, 2.4, 8.4, 7.9, 2.6, 0.1, 4.6, 7.6, -0.3, 3.8, 1.6, 0.9, 2.9, 7.9, 0.8, 7.5, 7.3, 8.5, 2.4, -0.3, 3.0, 3.5, 6.3, -2.2, 8.8, 2.1, 5.6, 11.7, 4.9, 5.7, 8.6, 3.3, 7.8, 1.7, 7.0, 6.8, 6.5, -3.7, 5.5, 9.7, 2.8, 5.5, 6.5, 7.2, 0.2, 2.3, 7.0, 11.0, 9.1, 5.2, 7.5, 3.5, 3.0, -0.7, 6.1, 10.3, 7.5, 4.1, 5.4, -2.6, 3.5, 9.1, 1.6, 10.1, 4.2, 6.1, 6.8, -0.9, 4.0, 10.4, 8.0, 5.5, 9.7, 0.7, 8.2, 3.9, 1.4, 8.2, 6.4, 5.2, -1.0, 5.5, 2.9, 4.2, 7.7, 7.1, 2.9, 5.9, 12.6, 3.4, 0.0, 8.1, 6.9, 7.2, 3.3, 10.0, 6.4, 10.9, 4.2, 4.4, 4.8, 5.7, 6.8, 5.6, 7.1, 7.8, 6.4, 8.5, 4.7, 6.5, 2.9, 7.3, 0.1, 5.5, 2.3, -0.2, 9.9, 4.3, -1.3, 7.2, 7.2, 4.5, 10.6, 9.7, 2.1, 2.5, 9.3, 6.7, 5.3, 3.9, 9.5, 6.4, 2.6, 2.7, 9.0, 6.7]))], ['x0', 'x1'], None), NumpyArray([6.3, 5.6, 4.2, 6.5, 5.8, 6.0, 9.3, 4.3, 2.6, 5.6, 2.7], [2, 1], [1, 1], 9)]), 5)], ['x0'], None)], ['x0'], None))

    for axis in range(5):
        print(" axis =", axis)
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
