################################################################ Contents

class Content:
    def __iter__(self):
        def convert(x):
            if isinstance(x, Content):
                return list(x)
            elif isinstance(x, tuple):
                return tuple(convert(y) for y in x)
            else:
                return x

        for i in range(len(self)):
            yield convert(self[i])

    def __repr__(self):
        return self.tostring_part("", "", "").rstrip()

class FloatArray(Content):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.data[where]
        else:
            return FloatArray(self.data[where])

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<FloatArray>\n"
        out += indent + "    " + " ".join(repr(x) for x in self.data) + "\n"
        out += indent + "</FloatArray>" + post
        return out

class ListArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Content)
        self.offsets = offsets
        self.content = content

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.content[self.offsets[where]:self.offsets[where + 1]]
        else:
            start = where.start
            stop = where.stop + 1
            return ListArray(self.offsets[start:stop], self.content)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<ListArray>\n"
        out += indent + "    <offsets>" + " ".join(repr(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListArray>" + post
        return out

class UnionArray(Content):
    def __init__(self, tags, offsets, contents):
        assert isinstance(tags, list)
        assert isinstance(offsets, list)
        assert all(isinstance(x, Content) for x in contents)
        self.tags = tags
        self.offsets = offsets
        self.contents = contents

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.contents[self.tags[where]][self.offsets[where]]
        else:
            return UnionArray(self.tags[where], self.offsets[where], self.contents)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<UnionArray>\n"
        out += indent + "    <tags>" + " ".join(repr(x) for x in self.tags) + "</tags>\n"
        out += indent + "    <offsets>" + " ".join(repr(x) for x in self.offsets) + "</offsets>\n"
        for i, content in enumerate(self.contents):
            out += content.tostring_part(indent + "    ", "<content i=\"{}\">".format(i), "</content>\n")
        out += indent + "</UnionArray>" + post
        return out

class OptionArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Content)
        self.offsets = offsets
        self.content = content

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, where):
        if isinstance(where, int):
            if self.offsets[where] == -1:
                return None
            else:
                return self.content[self.offsets[where]]
        else:
            return OptionArray(self.offsets[where], self.content)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<OptionArray>\n"
        out += indent + "    <offsets>" + " ".join(repr(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</OptionArray>" + post
        return out

class EmptyArray(Content):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, where):
        if isinstance(where, int):
            [][where]
        else:
            return EmptyArray()
    def tostring_part(self, indent, pre, post):
        return indent + pre + "<EmptyArray/>" + post

class TupleArray(Content):
    def __init__(self, contents):
        assert all(isinstance(x, Content) for x in contents)
        if len(contents) != 0:
            assert all(len(contents[0]) == len(x) for x in contents)
        self.contents = contents

    def __len__(self):
        if len(self.contents) == 0:
            return 0
        else:
            return len(self.contents[0])

    def __getitem__(self, where):
        if isinstance(where, int):
            return tuple(x[where] for x in self.contents)
        else:
            return TupleArray([x[where] for x in self.contents])

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<TupleArray>\n"
        for i, content in enumerate(self.contents):
            out += content.tostring_part(indent + "    ", "<content i=\"{}\">".format(i), "</content>\n")
        out += indent + "</TupleArray>" + post
        return out

################################################################ Content tests

one = OptionArray([0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6], UnionArray([1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3], [FloatArray([100, 200, 300]), ListArray([0, 1, 4, 4, 5], ListArray([0, 3, 3, 5, 6, 9], FloatArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])))]))
# print(one)
# print(list(one))
assert list(one) == [[[1.1, 2.2, 3.3]], None, 100, None, [[], [4.4, 5.5], [6.6]], None, 200, None, [], None, 300, None, [[7.7, 8.8, 9.9]]]
# print()

two = ListArray([0, 2, 2, 2, 3], TupleArray([FloatArray([1, 2, 3]), ListArray([0, 3, 3, 5], FloatArray([1.1, 2.2, 3.3, 4.4, 5.5]))]))
# print(two)
# print(list(two))
assert list(two) == [[(1, [1.1, 2.2, 3.3]), (2, [])], [], [], [(3, [4.4, 5.5])]]
# print()

################################################################ Fillables

class FillableArray:
    def __init__(self):
        self.fillable = UnknownFillable.fromempty()

    def fill(self, x):
        if x is None:
            self._maybeupdate(self.fillable.null())
        elif isinstance(x, (int, float)):
            self._maybeupdate(self.fillable.real(x))
        elif isinstance(x, list):
            self._maybeupdate(self.fillable.beginlist())
            for y in x:
                self.fill(y)
            self._maybeupdate(self.fillable.endlist())
        elif isinstance(x, tuple):
            self._maybeupdate(self.fillable.begintuple(len(x)))
            for i, y in enumerate(x):
                self.fillable.index(i)
                self.fill(y)
            self._maybeupdate(self.fillable.endtuple())
        else:
            raise AssertionError(x)

    def null(self):
        self._maybeupdate(self.fillable.null())

    def real(self, x):
        self._maybeupdate(self.fillable.real(x))

    def beginlist(self):
        self._maybeupdate(self.fillable.beginlist())

    def endlist(self):
        self._maybeupdate(self.fillable.endlist())

    def begintuple(self, numfields):
        self._maybeupdate(self.fillable.begintuple(numfields))

    def index(self, i):
        self._maybeupdate(self.fillable.index(i))

    def endtuple(self):
        self._maybeupdate(self.fillable.endtuple())

    def _maybeupdate(self, fillable):
        if fillable is not None and fillable is not self.fillable:
            self.fillable = fillable

    def snapshot(self):
        return self.fillable.snapshot()

class Fillable:
    pass

class UnknownFillable(Fillable):
    def __init__(self, nullcount):
        assert isinstance(nullcount, int)
        self.nullcount = nullcount

    @classmethod
    def fromempty(cls):
        return UnknownFillable(0)

    def snapshot(self):
        if self.nullcount == 0:
            return EmptyArray()
        else:
            return OptionArray([-1] * self.nullcount, EmptyArray())

    def __len__(self):
        return self.nullcount

    def null(self):
        self.nullcount += 1
        return self

    def real(self, x):
        if self.nullcount == 0:
            out = FloatFillable.fromempty()
        else:
            out = OptionFillable.fromnulls(self.nullcount, FloatFillable.fromempty())
        out.real(x)
        return out

    def beginlist(self):
        if self.nullcount == 0:
            out = ListFillable.fromempty()
        else:
            out = OptionFillable.fromnulls(self.nullcount, ListFillable.fromempty())
        out.beginlist()
        return out

    def endlist(self):
        return None

    def begintuple(self, numfields):
        out = TupleFillable.fromempty()
        out.begintuple(numfields)
        return out

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

class OptionFillable(Fillable):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Fillable)
        self.offsets = offsets
        self.content = content

    @classmethod
    def fromnulls(cls, nullcount, content):
        return cls([-1] * nullcount, content)

    @classmethod
    def fromvalids(cls, content):
        return cls(list(range(len(content))), content)

    def snapshot(self):
        return OptionArray(list(self.offsets), self.content.snapshot())

    def __len__(self):
        return len(self.offsets)

    def null(self):
        self.offsets.append(-1)
        return self

    def real(self, x):
        length = len(self.content)
        self._maybeupdate(self.content.real(x))
        if length != len(self.content):
            self.offsets.append(length)
        return self

    def beginlist(self):
        self._maybeupdate(self.content.beginlist())
        return self

    def endlist(self):
        length = len(self.content)
        if self.content.endlist() is None:
            return None
        else:
            self.offsets.append(length)
            return self

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

    def _maybeupdate(self, fillable):
        if fillable is not None and fillable is not self.content:
            self.content = content

class UnionFillable(Fillable):
    def __init__(self, tags, offsets, contents):
        assert isinstance(tags, list)
        assert isinstance(offsets, list)
        assert all(isinstance(x, Fillable) for x in contents)
        self.tags = tags
        self.offsets = offsets
        self.contents = contents

    @classmethod
    def fromsingle(cls, firstcontent):
        return UnionFillable([0] * len(firstcontent),
                             list(range(len(firstcontent))),
                             [firstcontent])

    def snapshot(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.tags)

    def null(self):
        raise NotImplementedError

    def real(self, x):
        raise NotImplementedError

    def beginlist(self):
        raise NotImplementedError

    def endlist(self):
        raise NotImplementedError

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

class ListFillable(Fillable):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Fillable)
        self.offsets = offsets
        self.content = content
        self.nextnested = False

    @classmethod
    def fromempty(cls):
        return ListFillable([0], UnknownFillable.fromempty())

    def snapshot(self):
        return ListArray(self.offsets, self.content.snapshot())

    def __len__(self):
        return len(self.offsets) - 1

    def null(self):
        if self.nextnested:
            self._maybeupdate(self.content.null())
            return self
        else:
            out = OptionFillable.fromvalids(self)
            out.null()
            return out

    def real(self, x):
        if self.nextnested:
            self._maybeupdate(self.content.real(x))
        else:
            raise NotImplementedError

    def beginlist(self):
        if self.nextnested:
            self._maybeupdate(self.content.beginlist())
            return self

        else:
            self.nextnested = True
            return self

    def endlist(self):
        if self.nextnested:
            if self.content.endlist() is None:
                self.offsets.append(len(self.content))
                self.nextnested = False
            return self

        else:
            return self

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

    def _maybeupdate(self, fillable):
        if fillable is not None and fillable is not self.content:
            self.content = fillable

class TupleFillable(Fillable):
    def __init__(self):
        self.length = -1

    @classmethod
    def fromempty(cls):
        return TupleFillable()

    def snapshot(self):
        return TupleArray([x.snapshot() for x in self.contents])

    def __len__(self):
        return self.length

    def null(self):
        raise NotImplementedError

    def real(self, x):
        assert self.nextindex != -1
        assert len(self.contents[self.nextindex]) == self.length
        self._maybeupdate(self.nextindex, self.contents[self.nextindex].real(x))
        return self

    def beginlist(self):
        raise NotImplementedError

    def endlist(self):
        raise NotImplementedError

    def begintuple(self, numfields):
        if self.length == -1:
            self.contents = [UnknownFillable.fromempty() for i in range(numfields)]
            self.length = 0
            self.nextindex = -1
            self.nextnested = False
            return self

        elif self.nextnested:
            assert self.nextindex != -1
            self._maybeupdate(self.nextindex, self.contents[self.nextindex].begintuple(numfields))
            return self

        elif self.nextindex != -1:
            assert len(self.contents[self.nextindex]) == self.length
            self._maybeupdate(self.nextindex, self.contents[self.nextindex].begintuple(numfields))
            self.nextnested = True
            return self

        elif len(self.contents) == numfields:
            return self

        else:
            raise NotImplementedError

    def index(self, i):
        if self.nextnested:
            assert self.nextindex != -1
            self._maybeupdate(self.nextindex, self.contents[self.nextindex].index(i))
            return self

        else:
            self.nextindex = i
            return self

    def endtuple(self):
        if self.nextnested:
            assert self.nextindex != -1
            self._maybeupdate(self.nextindex, self.contents[self.nextindex].endtuple())
            if len(self.contents[self.nextindex]) == self.length + 1:
                self.nextnested = False
            return self

        else:
            for i in range(len(self.contents)):
                if len(self.contents[i]) == self.length:
                    self._maybeupdate(i, self.contents[i].null())
                assert len(self.contents[i]) == self.length + 1
            self.length += 1
            self.nextindex = -1
            return self

    def _maybeupdate(self, index, fillable):
        if fillable is not None and fillable is not self.contents[index]:
            self.contents[index] = fillable

class FloatFillable(Fillable):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data

    @classmethod
    def fromempty(cls):
        return FloatFillable([])

    def snapshot(self):
        return FloatArray(list(self.data))

    def __len__(self):
        return len(self.data)

    def null(self):
        out = OptionFillable.fromvalids(self)
        out.null()
        return out

    def real(self, x):
        self.data.append(x)

    def beginlist(self):
        raise NotImplementedError

    def endlist(self):
        return None

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

################################################################ Fillable tests

fillable = FillableArray()
assert list(fillable.snapshot()) == []
fillable.begintuple(2)
fillable.index(0)
fillable.real(1.1)
fillable.endtuple()
assert list(fillable.snapshot()) == [(1.1, None)]
fillable.begintuple(2)
fillable.index(0)
fillable.real(2.2)
fillable.endtuple()
assert list(fillable.snapshot()) == [(1.1, None), (2.2, None)]
fillable.begintuple(2)
fillable.index(1)
fillable.real(3.3)
fillable.endtuple()
assert list(fillable.snapshot()) == [(1.1, None), (2.2, None), (None, 3.3)]

datasets = [
    [],
    [None],
    [None, None, None],
    [1.1, 2.2, 3.3],
    [None, 1.1, 2.2, 3.3],
    [1.1, None, 2.2, 3.3],
    [None, 1.1, None, 2.2, 3.3],
    [1.1, 2.2, 3.3, None],
    [1.1, 2.2, None, 3.3, None],
    [None, 1.1, 2.2, 3.3, None],
    [(1, 1.1), (2, 2.2), (3, 3.3)],
    [(1, (2, 3)), (10, (20, 30)), (100, (200, 300))],
    [(1, (2, 3, 4)), (10, (20, 30, 40)), (100, (200, 300, 400))],
    [((1, 2), (3, 4)), ((10, 20), (30, 40)), ((100, 200), (300, 400))],
    [((1, 2, 3), (4, 5)), ((10, 20, 30), (40, 50)), ((100, 200, 300), (400, 500))],
    [((1, 2, 3), (4, 5, 6)), ((10, 20, 30), (40, 50, 60)), ((100, 200, 300), (400, 500, 600))],
    [(1, (2, (3, 4))), (10, (20, (30, 40))), (100, (200, (300, 400)))],
    [(1, ((2, 3), 4)), (10, ((20, 30), 40)), (100, ((200, 300), 400))],
    [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3]],
    [None, [1.1], [1.1, 2.2], [1.1, 2.2, 3.3]],
    [[1.1], None, [1.1, 2.2], [1.1, 2.2, 3.3]],
    [None, [1.1], None, [1.1, 2.2], [1.1, 2.2, 3.3]],
    [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], None],
    [[1.1], None, [1.1, 2.2], [1.1, 2.2, 3.3], None],
    [None, [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], None],
    [[1.1], [1.1, 2.2], [1.1, None, 3.3]],
    [None, [1.1], [1.1, 2.2], [1.1, None, 3.3]],

    ]

for dataset in datasets:
    fillable = FillableArray()
    for x in dataset:
        fillable.fill(x)
    if list(fillable.snapshot()) != dataset:
        print(dataset)
        print(list(fillable.snapshot()))
        raise AssertionError

# fillable = FillableArray()
# fillable.fill([1.1])
# fillable.fill([1.1, 2.2])
# fillable.fill([1.1, None, 3.3])
# print(fillable.snapshot())
# print(list(fillable.snapshot()))
