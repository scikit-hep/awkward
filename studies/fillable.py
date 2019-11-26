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

################################################################ Content tests

assert list(OptionArray([0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6], UnionArray([1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3], [FloatArray([100, 200, 300]), ListArray([0, 1, 4, 4, 5], ListArray([0, 3, 3, 5, 6, 9], FloatArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])))]))) == [[[1.1, 2.2, 3.3]], None, 100, None, [[], [4.4, 5.5], [6.6]], None, 200, None, [], None, 300, None, [[7.7, 8.8, 9.9]]]

assert list(ListArray([0, 2, 2, 2, 3], TupleArray([FloatArray([1, 2, 3]), ListArray([0, 3, 3, 5], FloatArray([1.1, 2.2, 3.3, 4.4, 5.5]))]))) == [[(1, [1.1, 2.2, 3.3]), (2, [])], [], [], [(3, [4.4, 5.5])]]

################################################################ Fillables

class Fillable:
    pass

class UnknownFillable(Fillable):
    def __init__(self, nullcount):
        assert isinstance(nullcount, int)

    @classmethod
    def fromempty(cls):
        return UnknownFillable(0)

    def __len__(self):
        return self.nullcount

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

    def __len__(self):
        return len(self.offsets)

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

    @classmethod
    def fromempty(cls):
        return ListFillable([0], UnknownFillable.fromempty())

    def __len__(self):
        return len(self.offsets) - 1

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

class TupleFillable(Fillable):
    def __init__(self, contents):
        assert all(isinstance(x, Fillable) for x in contents)
        self.contents = contents

    @classmethod
    def fromempty(cls):
        return TupleFillable([])

    def __len__(self):
        if len(self.contents) == 0:
            return 0
        else:
            return len(self.contents[0])

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

class FloatFillable(Fillable):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data

    @classmethod
    def fromempty(cls):
        return FloatFillable([])

    def __len__(self):
        return len(self.data)

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

################################################################ Fillable tests
