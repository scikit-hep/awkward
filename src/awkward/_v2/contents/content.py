# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1

np = ak.nplike.NumpyMetadata.instance()


class Content(object):
    def _init(self, identifier, parameters):
        if parameters is None:
            parameters = {}

        if identifier is not None and not isinstance(
            identifier, ak._v2.identifier.Identifier
        ):
            raise TypeError(
                "{0} 'identifier' must be an Identifier or None, not {1}".format(
                    type(self).__name__, repr(identifier)
                )
            )
        if not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be a dict or None, not {1}".format(
                    type(self).__name__, repr(identifier)
                )
            )

        self._identifier = identifier
        self._parameters = parameters

    @property
    def identifier(self):
        return self._identifier

    @property
    def parameters(self):
        return self._parameters

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    def _simplify_optiontype(self):
        return self

    def _simplify_uniontype(self):
        return self

    def _handle_error(self, error, slicer=None):
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")

            message = error.str.decode(errors="surrogateescape")

            if error.pass_through:
                raise ValueError(message + filename)

            else:
                if error.id != ak._util.kSliceNone and self._identifier is not None:
                    # FIXME https://github.com/scikit-hep/awkward-1.0/blob/45d59ef4ae45eebb02995b8e1acaac0d46fb9573/src/libawkward/util.cpp#L443-L450
                    pass

                if error.attempt != ak._util.kSliceNone:
                    message += " while attempting to get index {0}".format(
                        error.attempt
                    )

                message += filename

                if slicer is None:
                    raise ValueError(message)
                else:
                    raise NestedIndexError(self, slicer, message)

    def _headtail(self, oldtail):
        if len(oldtail) == 0:
            return (), ()
        else:
            return oldtail[0], oldtail[1:]

    def _getitem_next_field(self, head, tail, advanced):
        nexthead, nexttail = self._headtail(tail)
        return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

    def _getitem_next_fields(self, head, tail, advanced):
        only_fields, not_fields = [], []
        for x in tail:
            if ak._util.isstr(x) or isinstance(x, list):
                only_fields.append(x)
            else:
                not_fields.append(x)
        nexthead, nexttail = self._headtail(tuple(not_fields))
        return self._getitem_fields(head, tuple(only_fields))._getitem_next(
            nexthead, nexttail, advanced
        )

    def _getitem_next_newaxis(self, tail, advanced):
        nexthead, nexttail = self._headtail(tail)
        return ak._v2.contents.RegularArray(
            self._getitem_next(nexthead, nexttail, advanced),
            1,  # size
            0,  # zeros_length is irrelevant when the size is 1 (!= 0)
            None,
            None,
        )

    def _getitem_next_ellipsis(self, tail, advanced):
        mindepth, maxdepth = self.minmax_depth

        dimlength = sum(
            1 if isinstance(x, (int, slice, ak._v2.index.Index64)) else 0 for x in tail
        )

        if len(tail) == 0 or mindepth - 1 == maxdepth - 1 == dimlength:
            nexthead, nexttail = self._headtail(tail)
            return self._getitem_next(nexthead, nexttail, advanced)

        elif mindepth - 1 == dimlength or maxdepth - 1 == dimlength:
            raise NestedIndexError(
                self,
                Ellipsis,
                "ellipsis (`...`) can't be used on data with different numbers of dimensions",
            )

        else:
            return self._getitem_next(slice(None), (Ellipsis,) + tail, advanced)

    def _getitem_broadcast(self, items, nplike):
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
            elif isinstance(item, int):
                lookup.append(len(broadcastable))
                broadcastable.append(item)
            else:
                lookup.append(len(broadcastable))
                broadcastable.append(nplike.asarray(item))

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
                    elif issubclass(x.dtype.type, np.integer):
                        out.append(ak._v2.index.Index64(x.astype(np.int64).reshape(-1)))
                    elif issubclass(x.dtype.type, (np.bool_, bool)):
                        out.append(ak._v2.index.Index64(np.nonzero(x.reshape(-1))[0]))
                    else:
                        raise TypeError(
                            "array slice must be an array of integers or booleans, not\n\n    {0}".format(
                                repr(x).replace("\n", "\n    ")
                            )
                        )
                    out[-1].metadata["shape"] = x.shape

        return tuple(out)

    def __getitem__(self, where):
        try:
            if ak._util.isint(where):
                return self._getitem_at(where)

            elif isinstance(where, slice) and where.step is None:
                return self._getitem_range(where)

            elif isinstance(where, slice):
                return self.__getitem__((where,))

            elif ak._util.isstr(where):
                return self._getitem_field(where)

            elif where is np.newaxis:
                return self.__getitem__((where,))

            elif where is Ellipsis:
                return self.__getitem__((where,))

            elif isinstance(where, tuple):
                if len(where) == 0:
                    return self
                nextwhere = self._getitem_broadcast(
                    [self._prepare_tuple_item(x) for x in where],
                    self.nplike,
                )

                next = ak._v2.contents.RegularArray(self, len(self), 1, None, None)

                out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
                if len(out) == 0:
                    raise out._getitem_nothing()
                else:
                    return out._getitem_at(0)

            elif isinstance(where, ak.highlevel.Array):
                return self.__getitem__(where.layout)

            elif isinstance(where, ak.layout.Content):
                return self.__getitem__(v1_to_v2(where))

            elif isinstance(where, ak._v2.contents.numpyarray.NumpyArray):
                if issubclass(where.dtype.type, np.int64):
                    carry = ak._v2.index.Index64(where.data.reshape(-1))
                    allow_lazy = True
                elif issubclass(where.dtype.type, np.integer):
                    carry = ak._v2.index.Index64(
                        where.data.astype(np.int64).reshape(-1)
                    )
                    allow_lazy = "copied"  # True, but also can be modified in-place
                elif issubclass(where.dtype.type, (np.bool_, bool)):
                    carry = ak._v2.index.Index64(np.nonzero(where.data.reshape(-1))[0])
                    allow_lazy = "copied"  # True, but also can be modified in-place
                else:
                    raise TypeError(
                        "array slice must be an array of integers or booleans, not\n\n    {0}".format(
                            repr(where.data).replace("\n", "\n    ")
                        )
                    )
                out = self._getitem_next_array_wrap(
                    self._carry(carry, allow_lazy, NestedIndexError), where.shape
                )
                if len(out) == 0:
                    return out._getitem_nothing()
                else:
                    return out._getitem_at(0)

            elif isinstance(where, Content):
                return self.__getitem__((where,))

            elif isinstance(where, Iterable) and all(ak._util.isstr(x) for x in where):
                return self._getitem_fields(where)

            elif isinstance(where, Iterable):
                return self.__getitem__(
                    v1_to_v2(ak.operations.convert.to_layout(where))
                )

            else:
                raise TypeError(
                    "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                    "integer/boolean arrays (possibly with variable-length nested "
                    "lists or missing values), field name (str) or names (non-tuple "
                    "iterable of str) are valid indices for slicing, not\n\n    "
                    + repr(where).replace("\n", "\n    ")
                )

        except NestedIndexError as err:

            def format_slice(x):
                if isinstance(x, slice):
                    if x.step is None:
                        return "{0}:{1}".format(
                            "" if x.start is None else x.start,
                            "" if x.stop is None else x.stop,
                        )
                    else:
                        return "{0}:{1}:{2}".format(
                            "" if x.start is None else x.start,
                            "" if x.stop is None else x.stop,
                            x.step,
                        )
                elif isinstance(x, tuple):
                    return "(" + ", ".join(format_slice(y) for y in x) + ")"
                elif isinstance(x, ak._v2.index.Index64):
                    return str(x.data)
                elif isinstance(x, Content):
                    return str(ak.Array(v2_to_v1(x)))
                else:
                    return repr(x)

            raise IndexError(
                """cannot slice

    {0}

with

    {1}

at inner {2} of length {3}, using sub-slice {4}.{5}""".format(
                    repr(ak.Array(v2_to_v1(self))),
                    format_slice(where),
                    type(err.array).__name__,
                    len(err.array),
                    format_slice(err.slicer),
                    ""
                    if err.details is None
                    else "\n\n{0} error: {1}.".format(
                        type(err.array).__name__, err.details
                    ),
                )
            )

    def _prepare_tuple_item(self, item):
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
            return self._prepare_tuple_item(item.layout)

        elif isinstance(item, ak.layout.Content):
            return self._prepare_tuple_item(v1_to_v2(item))

        elif isinstance(item, ak._v2.contents.NumpyArray):
            return item.data

        elif isinstance(
            item,
            (
                ak._v2.contents.ListOffsetArray,
                ak._v2.contents.ListArray,
                ak._v2.contents.RegularArray,
            ),
        ):
            return item.toListOffsetArray64(False)

        elif isinstance(
            item,
            (
                ak._v2.contents.IndexedOptionArray,
                ak._v2.contents.ByteMaskedArray,
                ak._v2.contents.BitMaskedArray,
                ak._v2.contents.UnmaskedArray,
            ),
        ):
            return item.toIndexedOptionArray64()

        elif isinstance(item, Iterable) and all(ak._util.isstr(x) for x in item):
            return list(item)

        elif isinstance(item, Iterable):
            return self._prepare_tuple_item(
                v1_to_v2(ak.operations.convert.to_layout(item))
            )

        else:
            raise TypeError(
                "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                "integer/boolean arrays (possibly with variable-length nested "
                "lists or missing values), field name (str) or names (non-tuple "
                "iterable of str) are valid indices for slicing, not\n\n    "
                + repr(item).replace("\n", "\n    ")
            )

    def _getitem_next_array_wrap(self, outcontent, shape):
        length = shape[-2] if len(shape) >= 2 else 0
        out = ak._v2.contents.RegularArray(outcontent, shape[-1], length, None, None)

        for i in range(len(shape) - 2, -1, -1):
            length = shape[i - 1] if i > 0 else 0
            out = ak._v2.contents.RegularArray(out, shape[i], length, None, None)
        return out

    def _carry_asrange(self, carry):
        assert isinstance(carry, ak._v2.index.Index)

        result = carry.nplike.empty(1, dtype=np.bool_)
        self._handle_error(
            carry.nplike[
                "awkward_Index_iscontiguous",  # badly named
                np.bool_,
                carry.dtype.type,
            ](
                result,
                carry.to(carry.nplike),
                len(carry),
            )
        )
        if result[0]:
            if len(carry) == len(self):
                return self
            elif len(carry) < len(self):
                return self._getitem_range(slice(0, len(carry)))
            else:
                raise IndexError
        else:
            return None

    def _range_identifier(self, start, stop):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _field_identifier(self, field):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _fields_identifier(self, fields):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _carry_identifier(self, carry, exception):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _axis_wrap_if_negative(self, axis):
        if axis >= 0:
            return axis
        minmax = self.minmax_depth
        mindepth = minmax[0]
        maxdepth = minmax[1]
        depth = self.purelist_depth
        if mindepth == depth and maxdepth == depth:
            posaxis = depth + axis
            if posaxis < 0:
                raise np.AxisError(
                    "axis={0} exceeds the depth of this array ({1})".format(
                        axis, depth
                    )
                )
            return posaxis
        elif mindepth + axis == 0:
            raise np.AxisError(
                "axis={0} exceeds the depth of this array ({1})".format(
                    axis, depth
                )
            )
        return axis

    def _localindex_axis0(self):
        localindex = ak._v2.index.Index64.empty(len(self), self.nplike)
        self._handle_error(
            localindex.nplike["awkward_localindex", np.int64](
                localindex.to(localindex.nplike),
                len(localindex),
            )
        )
        return ak._v2.contents.NumpyArray(localindex)

    def localindex(self, axis):
        return self._localindex(axis, 0)

    @property
    def purelist_isregular(self):
        return self.Form.purelist_isregular.__get__(self)

    @property
    def purelist_depth(self):
        return self.Form.purelist_depth.__get__(self)

    @property
    def minmax_depth(self):
        return self.Form.minmax_depth.__get__(self)

    @property
    def branch_depth(self):
        return self.Form.branch_depth.__get__(self)


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
        return "cannot slice {0} with {1}{2}".format(
            type(self._array).__name__,
            repr(self._slicer),
            "" if self._details is None else ": " + self._details,
        )
