# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


class Lookup:
    def __init__(self, layout, generator=None):
        positions = []
        tolookup(layout, positions)

        def arrayptr(x):
            if isinstance(x, int):
                return x
            else:
                return x.ctypes.data

        self.nplike = layout.backend.nplike
        self.generator = generator
        self.positions = positions
        self.arrayptrs = self.nplike.array(
            [arrayptr(x) for x in positions], dtype=np.intp
        )


def tolookup(layout, positions):
    if isinstance(layout, ak.contents.EmptyArray):
        return tolookup(layout.to_NumpyArray(np.dtype(np.float64)), positions)

    elif isinstance(layout, ak.contents.NumpyArray):
        if len(layout.shape) == 1:
            return NumpyLookup.tolookup(layout, positions)
        else:
            return tolookup(layout.to_RegularArray(), positions)

    elif isinstance(layout, ak.contents.RegularArray):
        return RegularLookup.tolookup(layout, positions)

    elif isinstance(layout, (ak.contents.ListArray, ak.contents.ListOffsetArray)):
        return ListLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.IndexedArray):
        return IndexedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.IndexedOptionArray):
        return IndexedOptionLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.ByteMaskedArray):
        return ByteMaskedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.BitMaskedArray):
        return BitMaskedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.UnmaskedArray):
        return UnmaskedLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.RecordArray):
        return RecordLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.record.Record):
        return RecordLookup.tolookup(layout, positions)

    elif isinstance(layout, ak.contents.UnionArray):
        return UnionLookup.tolookup(layout, positions)

    else:
        raise ak._errors.wrap_error(
            AssertionError(f"unrecognized Content: {type(layout)}")
        )


class ContentLookup:
    pass


class NumpyLookup(ContentLookup):
    ARRAY = 0

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(layout.to_contiguous().data)
        return pos

    def tolayout(self, lookup, pos, fields):
        assert fields == ()
        return ak.contents.NumpyArray(
            lookup.positions[pos + self.ARRAY], parameters=self.parameters
        )


class RegularLookup(ContentLookup):
    ZEROS_LENGTH = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(len(layout))
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.RegularArray(
            content,
            self.size,
            lookup.positions[pos + self.ZEROS_LENGTH],
            parameters=self.parameters,
        )


class ListLookup(ContentLookup):
    STARTS = 0
    STOPS = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(layout.starts.data)
        positions.append(layout.stops.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        starts = self.IndexOf(self.indextype)(lookup.positions[pos + self.STARTS])
        stops = self.IndexOf(self.indextype)(lookup.positions[pos + self.STOPS])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.ListArray(starts, stops, content, parameters=self.parameters)


class IndexedLookup(ContentLookup):
    INDEX = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(layout.index.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        index = self.IndexOf(self.indextype)(lookup.positions[pos + self.INDEX])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.IndexedArray(index, content, parameters=self.parameters)


class IndexedOptionLookup(ContentLookup):
    INDEX = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(layout.index.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        index = self.IndexOf(self.indextype)(lookup.positions[pos + self.INDEX])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.IndexedOptionArray(
            index, content, parameters=self.parameters
        )


class ByteMaskedLookup(ContentLookup):
    MASK = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(layout.mask.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        mask = self.IndexOf(self.masktype)(lookup.positions[pos + self.MASK])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.ByteMaskedArray(
            mask, content, self.valid_when, parameters=self.parameters
        )


class BitMaskedLookup(ContentLookup):
    LENGTH = 0
    MASK = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(len(layout))
        positions.append(layout.mask.data)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        mask = self.IndexOf(self.masktype)(lookup.positions[pos + self.MASK])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.BitMaskedArray(
            mask,
            content,
            self.valid_when,
            lookup.positions[pos + self.LENGTH],
            self.lsb_order,
            parameters=self.parameters,
        )


class UnmaskedLookup(ContentLookup):
    CONTENT = 0

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(None)
        positions[pos + cls.CONTENT] = tolookup(layout.content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.contents.UnmaskedArray(content, parameters=self.parameters)


class RecordLookup(ContentLookup):
    LENGTH = 0
    CONTENTS = 1

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(len(layout))
        positions.extend([None] * len(layout.contents))
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = tolookup(content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
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

            return ak.contents.RecordArray(
                contents,
                self.fields,
                lookup.positions[pos + self.LENGTH],
                parameters=self.parameters,
            )


class UnionLookup(ContentLookup):
    TAGS = 0
    INDEX = 1
    CONTENTS = 2

    @classmethod
    def tolookup(cls, layout, positions):
        pos = len(positions)
        positions.append(layout.tags.data)
        positions.append(layout.index.data)
        positions.extend([None] * len(layout.contents))
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = tolookup(content, positions)
        return pos

    def tolayout(self, lookup, pos, fields):
        tags = self.IndexOf(self.tagstype)(lookup.positions[pos + self.TAGS])
        index = self.IndexOf(self.indextype)(lookup.positions[pos + self.INDEX])
        contents = []
        for i, contenttype in enumerate(self.contenttypes):
            layout = contenttype.tolayout(
                lookup, lookup.positions[pos + self.CONTENTS + i], fields
            )
            contents.append(layout)
        return ak.contents.UnionArray(tags, index, contents, parameters=self.parameters)
