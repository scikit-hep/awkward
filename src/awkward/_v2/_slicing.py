# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

import awkward as ak
from awkward._v2.tmp_for_testing import v1_to_v2

np = ak.nplike.NumpyMetadata.instance()


def headtail(oldtail):
    if len(oldtail) == 0:
        return (), ()
    else:
        return oldtail[0], oldtail[1:]


def getitem_broadcast(items):
    lookup = []
    broadcastable = []
    for item in items:
        if (
            isinstance(
                item,
                (
                    slice,
                    list,
                    ak._v2.contents.ListOffsetArray,
                    ak._v2.contents.IndexedOptionArray,
                ),
            )
            or ak._util.isstr(item)
            or item is np.newaxis
            or item is Ellipsis
        ):
            lookup.append(None)
        else:
            lookup.append(len(broadcastable))
            broadcastable.append(item)

    nplike = ak.nplike.of(*broadcastable)

    broadcasted = nplike.broadcast_arrays(*broadcastable)

    out = []
    for i, item in zip(lookup, items):
        if i is None:
            out.append(item)
        else:
            x = broadcasted[i]
            if len(x.shape) == 0:
                out.append(int(x))
            else:
                if issubclass(x.dtype.type, np.int64):
                    out.append(ak._v2.index.Index64(x.reshape(-1)))
                    out[-1].metadata["shape"] = x.shape
                elif issubclass(x.dtype.type, np.integer):
                    out.append(ak._v2.index.Index64(x.astype(np.int64).reshape(-1)))
                    out[-1].metadata["shape"] = x.shape
                elif issubclass(x.dtype.type, (np.bool_, bool)):
                    if len(x.shape) == 1:
                        out.append(ak._v2.index.Index64(nplike.nonzero(x)[0]))
                        out[-1].metadata["shape"] = x.shape
                    else:
                        for w in nplike.nonzero(x):
                            out.append(ak._v2.index.Index64(w))
                else:
                    raise TypeError(
                        "array slice must be an array of integers or booleans, not\n\n    {}".format(
                            repr(x).replace("\n", "\n    ")
                        )
                    )

    return tuple(out)


def prepare_tuple_item(item):
    if ak._util.isint(item):
        return int(item)

    elif isinstance(item, slice):
        return item

    elif ak._util.isstr(item):
        return item

    elif item is np.newaxis:
        return item

    elif item is Ellipsis:
        return item

    elif isinstance(item, ak.highlevel.Array):
        return prepare_tuple_item(item.layout)

    elif isinstance(item, ak.layout.Content):
        return prepare_tuple_item(v1_to_v2(item))

    elif isinstance(item, ak._v2.highlevel.Array):
        return prepare_tuple_item(item.layout)

    elif isinstance(item, ak._v2.contents.EmptyArray):
        return prepare_tuple_item(item.toNumpyArray(np.int64))

    elif isinstance(item, ak._v2.contents.NumpyArray):
        return item.data

    elif isinstance(item, ak._v2.contents.Content):
        out = prepare_tuple_bool_to_int(prepare_tuple_nested(item))
        if isinstance(out, ak._v2.contents.NumpyArray):
            return out.data
        else:
            return out

    elif isinstance(item, Iterable) and all(ak._util.isstr(x) for x in item):
        return list(item)

    elif isinstance(item, Iterable):
        layout = ak._v2.operations.convert.to_layout(item)
        as_array = layout.maybe_to_array(layout.nplike)
        if as_array is None:
            return prepare_tuple_item(layout)
        else:
            return as_array

    else:
        raise TypeError(
            "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
            "integer/boolean arrays (possibly with variable-length nested "
            "lists or missing values), field name (str) or names (non-tuple "
            "iterable of str) are valid indices for slicing, not\n\n    "
            + repr(item).replace("\n", "\n    ")
        )


def prepare_tuple_RegularArray_toListOffsetArray64(item):
    if isinstance(item, ak._v2.contents.RegularArray):

        next = item.toListOffsetArray64()
        return ak._v2.contents.ListOffsetArray(
            next.offsets,
            prepare_tuple_RegularArray_toListOffsetArray64(next.content),
            identifier=item.identifier,
            parameters=item.parameters,
        )

    elif isinstance(item, ak._v2.contents.NumpyArray):
        return item

    else:
        raise AssertionError(type(item))


def prepare_tuple_nested(item):
    if isinstance(item, ak._v2.contents.EmptyArray):
        return prepare_tuple_nested(item.toNumpyArray(np.int64))

    elif isinstance(item, ak._v2.contents.NumpyArray) and issubclass(
        item.dtype.type, (bool, np.bool_, np.integer)
    ):
        if issubclass(item.dtype.type, (bool, np.bool_, np.int64)):
            next = item
        else:
            next = ak._v2.contents.NumpyArray(
                item.data.astype(np.int64),
                identifier=item.identifier,
                parameters=item.parameters,
                nplike=item.nplike,
            )
        next = next.toRegularArray()
        next = prepare_tuple_RegularArray_toListOffsetArray64(next)
        return next

    elif isinstance(
        item,
        ak._v2.contents.ListOffsetArray,
    ) and issubclass(item.offsets.dtype.type, np.int64):
        return ak._v2.contents.ListOffsetArray(
            item.offsets,
            prepare_tuple_nested(item.content),
            identifier=item.identifier,
            parameters=item.parameters,
        )

    elif isinstance(
        item,
        (
            ak._v2.contents.ListOffsetArray,
            ak._v2.contents.ListArray,
            ak._v2.contents.RegularArray,
        ),
    ):
        next = item.toListOffsetArray64(False)
        return prepare_tuple_nested(next)

    elif isinstance(
        item,
        (
            ak._v2.contents.IndexedArray,
            ak._v2.contents.IndexedOptionArray,
            ak._v2.contents.ByteMaskedArray,
            ak._v2.contents.BitMaskedArray,
            ak._v2.contents.UnmaskedArray,
        ),
    ) and isinstance(
        item.content,
        (
            ak._v2.contents.IndexedArray,
            ak._v2.contents.IndexedOptionArray,
            ak._v2.contents.ByteMaskedArray,
            ak._v2.contents.BitMaskedArray,
            ak._v2.contents.UnmaskedArray,
        ),
    ):
        # FIXME: might infinite-loop before simplify_optiontype is implemented
        next = item.simplify_optiontype()
        return prepare_tuple_nested(next)

    elif isinstance(
        item,
        ak._v2.contents.IndexedArray,
    ):
        next = item.project()
        return prepare_tuple_nested(next)

    elif isinstance(
        item,
        ak._v2.contents.IndexedOptionArray,
    ):
        nextindex = item.index.data.astype(np.int64)  # this ALWAYS copies
        nonnull = nextindex >= 0

        projected = item.content._carry(
            ak._v2.index.Index64(nextindex[nonnull]), False, NestedIndexError
        )

        # content has been projected; index must agree
        nextindex[nonnull] = item.nplike.arange(projected.length, dtype=np.int64)

        return ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(nextindex),
            prepare_tuple_nested(projected),
            identifier=item.identifier,
            parameters=item.parameters,
        )

    elif isinstance(
        item,
        (
            ak._v2.contents.ByteMaskedArray,
            ak._v2.contents.BitMaskedArray,
            ak._v2.contents.UnmaskedArray,
        ),
    ):
        is_valid = item.mask_as_bool(valid_when=True)
        positions_where_valid = item.nplike.nonzero(is_valid)[0]

        nextcontent = prepare_tuple_nested(item.content)._carry(
            ak._v2.index.Index64(positions_where_valid), False, NestedIndexError
        )

        nextindex = item.nplike.full(is_valid.shape[0], -1, np.int64)
        nextindex[positions_where_valid] = item.nplike.arange(
            positions_where_valid.shape[0], dtype=np.int64
        )

        return ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(nextindex),
            nextcontent,
            identifier=item.identifier,
            parameters=item.parameters,
        )

    elif isinstance(item, ak._v2.contents.UnionArray):
        # needs simplify_uniontype
        raise NotImplementedError("FIXME: need to implement UnionArray as a slice")

    else:
        raise TypeError(
            "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
            "integer/boolean arrays (possibly with variable-length nested "
            "lists or missing values), field name (str) or names (non-tuple "
            "iterable of str) are valid indices for slicing, not\n\n    "
            + repr(item).replace("\n", "\n    ")
        )


def prepare_tuple_bool_to_int(item):
    if (
        isinstance(item, ak._v2.contents.ListOffsetArray)
        and isinstance(item.content, ak._v2.contents.NumpyArray)
        and issubclass(item.content.dtype.type, (bool, np.bool_))
    ):
        if item.nplike.known_data or item.nplike.known_shape:
            localindex = item.localindex(axis=1)
            nextcontent = localindex.content.data[item.content.data]

            cumsum = item.nplike.empty(item.content.data.shape[0] + 1, np.int64)
            cumsum[0] = 0
            cumsum[1:] = item.nplike.cumsum(item.content.data)
            nextoffsets = cumsum[item.offsets]

        else:
            nextoffsets = item.offsets
            nextcontent = item.nplike.empty(
                (ak._v2._typetracer.UnknownLength,), dtype=np.int64
            )

        return ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(nextoffsets),
            ak._v2.contents.NumpyArray(nextcontent, nplike=item.nplike),
        )

    elif (
        isinstance(item, ak._v2.contents.ListOffsetArray)
        and isinstance(item.content, ak._v2.contents.IndexedOptionArray)
        and isinstance(item.content.content, ak._v2.contents.NumpyArray)
        and issubclass(item.content.content.dtype.type, (bool, np.bool_))
    ):
        if item.nplike.known_data or item.nplike.known_shape:
            # missing values as any integer other than -1 are extremely rare
            isnegative = item.content.index.data < 0
            if item.nplike.any(item.content.index.data < -1):
                safeindex = item.content.index.data.copy()
                safeindex[isnegative] = -1
            else:
                safeindex = item.content.index.data

            # expanded is a new buffer (can be modified in-place)
            if item.content.content.data.shape[0] > 0:
                expanded = item.content.content.data[safeindex]
            else:
                expanded = item.content.content.data.nplike.ones(
                    safeindex.shape[0], np.bool_
                )

            localindex = item.localindex(axis=1)

            # nextcontent does not include missing values
            expanded[isnegative] = False
            nextcontent = localindex.content.data[expanded]

            # list offsets do include missing values
            expanded[isnegative] = True
            cumsum = item.nplike.empty(expanded.shape[0] + 1, np.int64)
            cumsum[0] = 0
            cumsum[1:] = item.nplike.cumsum(expanded)
            nextoffsets = cumsum[item.offsets]

            # outindex fits into the lists; non-missing are sequential
            outindex = item.nplike.full(nextoffsets[-1], -1, np.int64)
            outindex[~isnegative[expanded]] = item.nplike.arange(
                nextcontent.shape[0], dtype=np.int64
            )

        else:
            nextoffsets = item.offsets
            outindex = item.content.index
            nextcontent = item.nplike.empty(
                (ak._v2._typetracer.UnknownLength,), dtype=np.int64
            )

        return ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(nextoffsets),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index(outindex),
                ak._v2.contents.NumpyArray(nextcontent, nplike=item.nplike),
            ),
        )

    elif isinstance(item, ak._v2.contents.ListOffsetArray):
        return ak._v2.contents.ListOffsetArray(
            item.offsets, prepare_tuple_bool_to_int(item.content)
        )

    elif isinstance(item, ak._v2.contents.IndexedOptionArray):
        if isinstance(item.content, ak._v2.contents.ListOffsetArray):
            return ak._v2.contents.IndexedOptionArray(
                item.index, prepare_tuple_bool_to_int(item.content)
            )

        if isinstance(item.content, ak._v2.contents.NumpyArray) and issubclass(
            item.content.dtype.type, (bool, np.bool_)
        ):
            if item.nplike.known_data or item.nplike.known_shape:
                # missing values as any integer other than -1 are extremely rare
                isnegative = item.index.data < 0
                if item.nplike.any(item.index.data < -1):
                    safeindex = item.index.data.copy()
                    safeindex[isnegative] = -1
                else:
                    safeindex = item.index.data

                # expanded is a new buffer (can be modified in-place)
                if item.content.data.shape[0] > 0:
                    expanded = item.content.data[safeindex]
                else:
                    expanded = item.content.data.nplike.ones(
                        safeindex.shape[0], np.bool_
                    )

                # nextcontent does not include missing values
                expanded[isnegative] = False
                nextcontent = item.nplike.nonzero(expanded)[0]

                # outindex does include missing values
                expanded[isnegative] = True
                lenoutindex = item.nplike.count_nonzero(expanded)

                # non-missing are sequential
                outindex = item.nplike.full(lenoutindex, -1, np.int64)
                outindex[~isnegative[expanded]] = item.nplike.arange(
                    nextcontent.shape[0], dtype=np.int64
                )

            else:
                outindex = item.index
                nextcontent = item.nplike.empty(
                    (ak._v2._typetracer.UnknownLength,), dtype=np.int64
                )

            return ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index(outindex),
                ak._v2.contents.NumpyArray(nextcontent, nplike=item.nplike),
            )

        else:
            return ak._v2.contents.IndexedOptionArray(
                item.index, prepare_tuple_bool_to_int(item.content)
            )

    elif isinstance(item, ak._v2.contents.NumpyArray):
        assert item.data.shape == (item.length,)
        return item

    else:
        raise AssertionError(type(item))


def getitem_next_array_wrap(outcontent, shape):
    length = shape[-2] if len(shape) >= 2 else 0
    size = shape[-1]
    if isinstance(size, ak._v2._typetracer.UnknownLengthType):
        size = 1
    out = ak._v2.contents.RegularArray(outcontent, size, length, None, None)

    for i in range(len(shape) - 2, -1, -1):
        length = shape[i - 1] if i > 0 else 0
        size = shape[i]
        if isinstance(size, ak._v2._typetracer.UnknownLengthType):
            size = 1
        out = ak._v2.contents.RegularArray(out, size, length, None, None)
    return out


class NestedIndexError(IndexError):
    def __init__(self, array, slicer, details=None):
        self._array = array
        self._slicer = slicer
        self._details = details

    @property
    def array(self):
        return self._array

    @property
    def slicer(self):
        return self._slicer

    @property
    def details(self):
        return self._details

    def __str__(self):
        return "cannot slice {} with {}{}".format(
            type(self._array).__name__,
            repr(self._slicer),
            "" if self._details is None else ": " + self._details,
        )
