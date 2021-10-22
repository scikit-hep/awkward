# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
import numpy as np


def v1v2_id_equal(v1, v2):
    if v1 is None and v2 is None:
        return True
    else:
        raise NotImplementedError("Identities/Identifier equality")


def v1v2_equal(v1, v2):
    assert isinstance(v1, ak.layout.Content) and isinstance(v2, ak._v2.contents.Content)

    if len(v1) != len(v2):
        return False

    if not v1v2_id_equal(v1.identities, v2.identifier):
        return False

    if v1.parameters != v2.parameters:
        return False

    if isinstance(v1, ak.layout.EmptyArray) and isinstance(
        v2, ak._v2.contents.EmptyArray
    ):
        return True

    elif isinstance(v1, ak.layout.NumpyArray) and isinstance(
        v2, ak._v2.contents.NumpyArray
    ):
        v1_data = np.asarray(v1)
        return v1_data.dtype == v2.dtype and np.array_equal(v1_data, v2.data)

    elif isinstance(v1, ak.layout.RegularArray) and isinstance(
        v2, ak._v2.contents.RegularArray
    ):
        return v1v2_equal(v1.content, v2.content) and v1.size == v2.size

    elif isinstance(
        v1, (ak.layout.ListArray32, ak.layout.ListArrayU32, ak.layout.ListArray64)
    ) and isinstance(v2, ak._v2.contents.ListArray):
        v1_starts = np.asarray(v1.starts)
        v1_stops = np.asarray(v1.stops)
        v2_starts = v2.starts.data
        v2_stops = v2.stops.data
        return (
            v1v2_equal(v1.content, v2.content)
            and np.array_equal(v1_starts, v2_starts)
            and v1_starts.dtype == v2_starts.dtype
            and np.array_equal(v1_stops, v2_stops)
            and v1_stops.dtype == v2_stops.dtype
        )

    elif (
        isinstance(
            v1,
            (
                ak.layout.ListOffsetArray32,
                ak.layout.ListOffsetArrayU32,
                ak.layout.ListOffsetArray64,
            ),
        )
        and isinstance(v2, ak._v2.contents.ListOffsetArray)
    ):
        v1_offsets = np.asarray(v1.offsets)
        v2_offsets = v2.offsets.data
        return (
            v1v2_equal(v1.content, v2.content)
            and np.array_equal(v1_offsets, v2_offsets)
            and v1_offsets.dtype == v2_offsets.dtype
        )

    elif isinstance(v1, ak.layout.RecordArray) and isinstance(
        v2, ak._v2.contents.RecordArray
    ):
        return v1.istuple == v2.is_tuple and all(
            v1v2_equal(v1.field(i)[: len(v1)], v2.content(i))
            for i in range(v1.numfields)
        )

    elif (
        isinstance(
            v1,
            (
                ak.layout.IndexedArray32,
                ak.layout.IndexedArrayU32,
                ak.layout.IndexedArray64,
            ),
        )
        and isinstance(v2, ak._v2.contents.IndexedArray)
    ):
        v1_index = np.asarray(v1.index)
        v2_index = v2.index.data
        return (
            v1v2_equal(v1.content, v2.content)
            and np.array_equal(v1_index, v2_index)
            and v1_index.dtype == v2_index.dtype
        )

    elif isinstance(
        v1, (ak.layout.IndexedOptionArray32, ak.layout.IndexedOptionArray64)
    ) and isinstance(v2, ak._v2.contents.IndexedOptionArray):
        v1_index = np.asarray(v1.index)
        v2_index = v2.index.data
        return (
            v1v2_equal(v1.content, v2.content)
            and np.array_equal(v1_index, v2_index)
            and v1_index.dtype == v2_index.dtype
        )

    elif isinstance(v1, ak.layout.ByteMaskedArray) and isinstance(
        v2, ak._v2.contents.ByteMaskedArray
    ):
        v1_mask = np.asarray(v1.mask)
        v2_mask = v2.mask.data
        return (
            v1v2_equal(v1.content, v2.content)
            and np.array_equal(v1_mask, v2_mask)
            and v1_mask.dtype == v2_mask.dtype
            and v1.valid_when == v2.valid_when
        )

    elif isinstance(v1, ak.layout.BitMaskedArray) and isinstance(
        v2, ak._v2.contents.BitMaskedArray
    ):
        v1_mask = np.asarray(v1.mask)
        v2_mask = v2.mask.data
        return (
            v1v2_equal(v1.content, v2.content)
            and np.array_equal(v1_mask, v2_mask)
            and v1_mask.dtype == v2_mask.dtype
            and v1.valid_when == v2.valid_when
            and v1.lsb_order == v2.lsb_order
        )

    elif isinstance(v1, ak.layout.UnmaskedArray) and isinstance(
        v2, ak._v2.contents.UnmaskedArray
    ):
        return v1v2_equal(v1.content, v2.content)

    elif (
        isinstance(
            v1,
            (
                ak.layout.UnionArray8_32,
                ak.layout.UnionArray8_U32,
                ak.layout.UnionArray8_64,
            ),
        )
        and isinstance(v2, ak._v2.contents.UnionArray)
    ):
        v1_tags = np.asarray(v1.tags)
        v2_tags = v2.tags.data
        v1_index = np.asarray(v1.index)
        v2_index = v2.index.data
        return (
            np.array_equal(v1_tags, v2_tags)
            and v1_tags.dtype == v2_tags.dtype
            and np.array_equal(v1_index, v2_index)
            and v1_index.dtype == v2_index.dtype
            and all(
                v1v2_equal(v1.content(i), v2.content(i)) for i in range(v1.numcontents)
            )
        )

    elif isinstance(v1, ak.layout.VirtualArray) and isinstance(
        v2, ak._v2.contents.VirtualArray
    ):
        raise AssertionError("VirtualArray is v1 only")

    else:
        raise AssertionError("{0} vs {1}".format(type(v1), type(v2)))


def v1_to_v2_id(v1):
    if v1 is None:
        return None
    else:
        raise NotImplementedError("Identities to Identifier")


def fix(array):
    if issubclass(array.dtype.type, np.signedinteger):
        if array.dtype.itemsize == 8:
            return array.view(np.int64)
        elif array.dtype.itemsize == 4:
            return array.view(np.int32)
        elif array.dtype.itemsize == 2:
            return array.view(np.int16)
        elif array.dtype.itemsize == 1:
            return array.view(np.int8)
    elif issubclass(array.dtype.type, np.unsignedinteger):
        if array.dtype.itemsize == 8:
            return array.view(np.uint64)
        elif array.dtype.itemsize == 4:
            return array.view(np.uint32)
        elif array.dtype.itemsize == 2:
            return array.view(np.uint16)
        elif array.dtype.itemsize == 1:
            return array.view(np.uint8)
    else:
        return array


def v1_to_v2(v1):
    assert isinstance(v1, ak.layout.Content)

    if isinstance(v1, ak.layout.EmptyArray):
        return ak._v2.contents.EmptyArray(
            identifier=v1_to_v2_id(v1.identities), parameters=v1.parameters
        )

    elif isinstance(v1, ak.layout.NumpyArray):
        primitive = v1.form.primitive
        if primitive == "datetime64" or primitive == "timedelta64":
            return ak._v2.contents.NumpyArray(
                fix(np.asarray(v1.view_int64)).view(np.dtype(v1.format)),
                identifier=v1_to_v2_id(v1.identities),
                parameters=v1.parameters,
            )
        else:
            return ak._v2.contents.NumpyArray(
                fix(np.asarray(v1)),
                identifier=v1_to_v2_id(v1.identities),
                parameters=v1.parameters,
            )

    elif isinstance(v1, ak.layout.RegularArray):
        return ak._v2.contents.RegularArray(
            v1_to_v2(v1.content),
            v1.size,
            len(v1),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(
        v1, (ak.layout.ListArray32, ak.layout.ListArrayU32, ak.layout.ListArray64)
    ):
        return ak._v2.contents.ListArray(
            ak._v2.index.Index(fix(np.asarray(v1.starts))),
            ak._v2.index.Index(fix(np.asarray(v1.stops))),
            v1_to_v2(v1.content),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(
        v1,
        (
            ak.layout.ListOffsetArray32,
            ak.layout.ListOffsetArrayU32,
            ak.layout.ListOffsetArray64,
        ),
    ):
        return ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index(fix(np.asarray(v1.offsets))),
            v1_to_v2(v1.content),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(v1, ak.layout.RecordArray):
        return ak._v2.contents.RecordArray(
            [v1_to_v2(x) for x in v1.contents],
            v1.recordlookup,
            len(v1),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(
        v1,
        (
            ak.layout.IndexedArray32,
            ak.layout.IndexedArrayU32,
            ak.layout.IndexedArray64,
        ),
    ):
        return ak._v2.contents.IndexedArray(
            ak._v2.index.Index(fix(np.asarray(v1.index))),
            v1_to_v2(v1.content),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(
        v1, (ak.layout.IndexedOptionArray32, ak.layout.IndexedOptionArray64)
    ):
        return ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index(fix(np.asarray(v1.index))),
            v1_to_v2(v1.content),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(v1, ak.layout.ByteMaskedArray):
        return ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index(fix(np.asarray(v1.mask))),
            v1_to_v2(v1.content),
            v1.valid_when,
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(v1, ak.layout.BitMaskedArray):
        return ak._v2.contents.BitMaskedArray(
            ak._v2.index.Index(fix(np.asarray(v1.mask))),
            v1_to_v2(v1.content),
            v1.valid_when,
            len(v1),
            v1.lsb_order,
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(v1, ak.layout.UnmaskedArray):
        return ak._v2.contents.UnmaskedArray(
            v1_to_v2(v1.content),
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(
        v1,
        (
            ak.layout.UnionArray8_32,
            ak.layout.UnionArray8_U32,
            ak.layout.UnionArray8_64,
        ),
    ):
        return ak._v2.contents.UnionArray(
            ak._v2.index.Index(fix(np.asarray(v1.tags))),
            ak._v2.index.Index(fix(np.asarray(v1.index))),
            [v1_to_v2(x) for x in v1.contents],
            identifier=v1_to_v2_id(v1.identities),
            parameters=v1.parameters,
        )

    elif isinstance(v1, ak.layout.VirtualArray):
        raise AssertionError("VirtualArray is v1 only")

    else:
        raise AssertionError(type(v1))


def v2_to_v1_id(v1):
    if v1 is None:
        return None
    else:
        raise NotImplementedError("Identifier to Identities")


def v2_to_v1(v2):
    assert isinstance(v2, ak._v2.contents.Content)

    if isinstance(v2, ak._v2.contents.EmptyArray):
        return ak.layout.EmptyArray(
            identities=v2_to_v1_id(v2.identifier), parameters=v2.parameters
        )

    elif isinstance(v2, ak._v2.contents.NumpyArray):
        return ak.layout.NumpyArray(
            v2.data, identities=v2_to_v1_id(v2.identifier), parameters=v2.parameters
        )

    elif isinstance(v2, ak._v2.contents.RegularArray):
        return ak.layout.RegularArray(
            v2_to_v1(v2.content),
            v2.size,
            len(v2),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.ListArray):
        assert v2.starts.dtype == v2.stops.dtype
        if v2.starts.dtype == np.dtype(np.int32):
            ind = ak.layout.Index32
            cls = ak.layout.ListArray32
        elif v2.starts.dtype == np.dtype(np.uint32):
            ind = ak.layout.IndexU32
            cls = ak.layout.ListArrayU32
        elif v2.starts.dtype == np.dtype(np.int64):
            ind = ak.layout.Index64
            cls = ak.layout.ListArray64
        return cls(
            ind(v2.starts.data),
            ind(v2.stops.data),
            v2_to_v1(v2.content),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.ListOffsetArray):
        if v2.offsets.dtype == np.dtype(np.int32):
            ind = ak.layout.Index32
            cls = ak.layout.ListOffsetArray32
        elif v2.offsets.dtype == np.dtype(np.uint32):
            ind = ak.layout.IndexU32
            cls = ak.layout.ListOffsetArrayU32
        elif v2.offsets.dtype == np.dtype(np.int64):
            ind = ak.layout.Index64
            cls = ak.layout.ListOffsetArray64
        return cls(
            ind(v2.offsets.data),
            v2_to_v1(v2.content),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.RecordArray):
        return ak.layout.RecordArray(
            [v2_to_v1(x) for x in v2.contents],
            None if v2.is_tuple else v2.fields,
            len(v2),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.IndexedArray):
        if v2.index.dtype == np.dtype(np.int32):
            ind = ak.layout.Index32
            cls = ak.layout.IndexedArray32
        elif v2.index.dtype == np.dtype(np.uint32):
            ind = ak.layout.IndexU32
            cls = ak.layout.IndexedArrayU32
        elif v2.index.dtype == np.dtype(np.int64):
            ind = ak.layout.Index64
            cls = ak.layout.IndexedArray64
        return cls(
            ind(v2.index.data),
            v2_to_v1(v2.content),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.IndexedOptionArray):
        if v2.index.dtype == np.dtype(np.int32):
            ind = ak.layout.Index32
            cls = ak.layout.IndexedOptionArray32
        elif v2.index.dtype == np.dtype(np.uint32):
            ind = ak.layout.IndexU32
            cls = ak.layout.IndexedOptionArrayU32
        elif v2.index.dtype == np.dtype(np.int64):
            ind = ak.layout.Index64
            cls = ak.layout.IndexedOptionArray64
        return cls(
            ind(v2.index.data),
            v2_to_v1(v2.content),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.ByteMaskedArray):
        return ak.layout.ByteMaskedArray(
            ak.layout.Index8(v2.mask.data),
            v2_to_v1(v2.content),
            v2.valid_when,
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.BitMaskedArray):
        return ak.layout.BitMaskedArray(
            ak.layout.IndexU8(v2.mask.data),
            v2_to_v1(v2.content),
            v2.valid_when,
            len(v2),
            v2.lsb_order,
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.UnmaskedArray):
        return ak.layout.UnmaskedArray(
            v2_to_v1(v2.content),
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    elif isinstance(v2, ak._v2.contents.UnionArray):
        if v2.index.dtype == np.dtype(np.int32):
            ind = ak.layout.Index32
            cls = ak.layout.UnionArray8_32
        elif v2.index.dtype == np.dtype(np.uint32):
            ind = ak.layout.IndexU32
            cls = ak.layout.UnionArray8_U32
        elif v2.index.dtype == np.dtype(np.int64):
            ind = ak.layout.Index64
            cls = ak.layout.UnionArray8_64
        return cls(
            ak.layout.Index8(v2.tags.data),
            ind(v2.index.data),
            [v2_to_v1(x) for x in v2.contents],
            identities=v2_to_v1_id(v2.identifier),
            parameters=v2.parameters,
        )

    else:
        raise AssertionError(type(v2))


def v1_to_v2_index(v1):
    assert isinstance(
        v1,
        (
            ak.layout.IndexU8,
            ak.layout.IndexU32,
            ak.layout.Index8,
            ak.layout.Index32,
            ak.layout.Index64,
        ),
    )

    if isinstance(v1, ak.layout.IndexU8):
        return ak._v2.index.IndexU8(np.asarray(v1))

    elif isinstance(v1, ak.layout.IndexU32):
        return ak._v2.index.IndexU32(np.asarray(v1))

    elif isinstance(v1, ak.layout.Index8):
        return ak._v2.index.Index8(np.asarray(v1))

    elif isinstance(v1, ak.layout.Index32):
        return ak._v2.index.Index32(np.asarray(v1))

    elif isinstance(v1, ak.layout.Index64):
        return ak._v2.index.Index64(np.asarray(v1))

    else:
        raise AssertionError(type(v1))
