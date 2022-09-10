# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Lookup:
    def __init__(self, layout, generator=None):
        positions = []
        tolookup(layout, positions)

        def arrayptr(x):
            if isinstance(x, int):
                return x
            else:
                return x.ctypes.data

        self.nplike = layout.nplike
        self.generator = generator
        self.positions = positions
        self.arrayptrs = self.nplike.array(
            [arrayptr(x) for x in positions], dtype=np.intp
        )


def tolookup(layout, positions):
    if isinstance(layout, ak._v2.contents.EmptyArray):
        return tolookup(layout.toNumpyArray(np.dtype(np.float64)), positions)

    elif isinstance(layout, ak._v2.contents.NumpyArray):
        if len(layout.shape) == 1:
            return NumpyLookup.tolookup(layout, positions)
        else:
            return tolookup(layout.toRegularArray(), positions)

    elif isinstance(layout, ak._v2.contents.RegularArray):
        return RegularLookup.tolookup(layout, positions)

    elif isinstance(
        layout, (ak._v2.contents.ListArray, ak._v2.contents.ListOffsetArray)
    ):
        return ListLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.IndexedArray):
        return IndexedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.IndexedOptionArray):
        return IndexedOptionLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.ByteMaskedArray):
        return ByteMaskedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.BitMaskedArray):
        return BitMaskedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.UnmaskedArray):
        return UnmaskedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.RecordArray):
        return RecordLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.record.Record):
        return RecordLookup.tolookup(layout, positions)

    elif isinstance(layout, ak._v2.contents.UnionArray):
        return UnionLookup.tolookup(layout, positions)

    else:
        raise ak._v2._util.error(
            AssertionError(f"unrecognized Content: {type(layout)}")
        )


class ContentLookup:
    @classmethod
    def tolookup_identifier(cls, layout, positions):
        if layout.identifier is None:
            positions.append(-1)
        else:
            positions.append(layout.identifier.data)


class NumpyLookup(ContentLookup):
    IDENTIFIER = 0
    ARRAY = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(layout.contiguous().data)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        assert fields == ()
        return ak._v2.contents.NumpyArray(
            lookup.positions[pos + self.ARRAY], parameters=self.parameters
        )


class RegularLookup(ContentLookup):
    IDENTIFIER = 0
    ZEROS_LENGTH = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(len(layout))
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.RegularArray(
            content,
            self.size,
            lookup.positions[pos + self.ZEROS_LENGTH],
            parameters=self.parameters,
        )


class ListLookup(ContentLookup):
    IDENTIFIER = 0
    STARTS = 1
    STOPS = 2
    CONTENT = 3

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(layout.starts.data)
        positions.append(layout.stops.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        starts = self.IndexOf(self.indextype)(lookup.positions[pos + self.STARTS])
        stops = self.IndexOf(self.indextype)(lookup.positions[pos + self.STOPS])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.ListArray(
            starts, stops, content, parameters=self.parameters
        )


class IndexedLookup(ContentLookup):
    IDENTIFIER = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(layout.index.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        index = self.IndexOf(self.indextype)(lookup.positions[pos + self.INDEX])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.IndexedArray(index, content, parameters=self.parameters)


class IndexedOptionLookup(ContentLookup):
    IDENTIFIER = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(layout.index.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        index = self.IndexOf(self.indextype)(lookup.positions[pos + self.INDEX])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.IndexedOptionArray(
            index, content, parameters=self.parameters
        )


class ByteMaskedLookup(ContentLookup):
    IDENTIFIER = 0
    MASK = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(layout.mask.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        mask = self.IndexOf(self.masktype)(lookup.positions[pos + self.MASK])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.ByteMaskedArray(
            mask, content, self.valid_when, parameters=self.parameters
        )


class BitMaskedLookup(ContentLookup):
    IDENTIFIER = 0
    LENGTH = 1
    MASK = 2
    CONTENT = 3

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(len(layout))
        positions.append(layout.mask.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        mask = self.IndexOf(self.masktype)(lookup.positions[pos + self.MASK])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.BitMaskedArray(
            mask,
            content,
            self.valid_when,
            lookup.positions[pos + self.LENGTH],
            self.lsb_order,
            parameters=self.parameters,
        )


class UnmaskedLookup(ContentLookup):
    IDENTIFIER = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak._v2.contents.UnmaskedArray(content, parameters=self.parameters)


class RecordLookup(ContentLookup):
    IDENTIFIER = 0
    LENGTH = 1
    CONTENTS = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(len(layout))
        positions.extend([None] * len(layout.contents))
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = tolookup(content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        if len(fields) > 0:
            index = self.fieldindex(fields[0])
            assert index is not None
            return self.contenttypes[index].tolayout(
                lookup, lookup.positions[pos + self.CONTENTS + index], fields[1:]
            )
        else:
            contents = []
            for i, contenttype in enumerate(self.contenttypes):
                layout = contenttype.tolayout(
                    lookup, lookup.positions[pos + self.CONTENTS + i], fields
                )
                contents.append(layout)

            return ak._v2.contents.RecordArray(
                contents,
                self.fields,
                lookup.positions[pos + self.LENGTH],
                parameters=self.parameters,
            )


class UnionLookup(ContentLookup):
    IDENTIFIER = 0
    TAGS = 1
    INDEX = 2
    CONTENTS = 3

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        cls.tolookup_identifier(layout, positions)
        positions.append(layout.tags.data)
        positions.append(layout.index.data)
        positions.extend([None] * len(layout.contents))
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = tolookup(content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert lookup.positions[pos + self.IDENTIFIER] == -1
        tags = self.IndexOf(self.tagstype)(lookup.positions[pos + self.TAGS])
        index = self.IndexOf(self.indextype)(lookup.positions[pos + self.INDEX])
        contents = []
        for i, contenttype in enumerate(self.contenttypes):
            layout = contenttype.tolayout(
                lookup, lookup.positions[pos + self.CONTENTS + i], fields
            )
            contents.append(layout)
        return ak._v2.contents.UnionArray(
            tags, index, contents, parameters=self.parameters
        )
