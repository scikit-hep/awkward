# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2._slicing import NestedIndexError
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

    def _simplify_uniontype(self):
        return self

    def _simplify_optiontype(self):
        return self

    def maybe_to_nplike(self, nplike):
        return None

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

    def _getitem_next_field(self, head, tail, advanced):
        nexthead, nexttail = ak._v2._slicing.headtail(tail)
        return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

    def _getitem_next_fields(self, head, tail, advanced):
        only_fields, not_fields = [], []
        for x in tail:
            if ak._util.isstr(x) or isinstance(x, list):
                only_fields.append(x)
            else:
                not_fields.append(x)
        nexthead, nexttail = ak._v2._slicing.headtail(tuple(not_fields))
        return self._getitem_fields(head, tuple(only_fields))._getitem_next(
            nexthead, nexttail, advanced
        )

    def _getitem_next_newaxis(self, tail, advanced):
        nexthead, nexttail = ak._v2._slicing.headtail(tail)
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
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            return self._getitem_next(nexthead, nexttail, advanced)

        elif mindepth - 1 == dimlength or maxdepth - 1 == dimlength:
            raise NestedIndexError(
                self,
                Ellipsis,
                "ellipsis (`...`) can't be used on data with different numbers of dimensions",
            )

        else:
            return self._getitem_next(slice(None), (Ellipsis,) + tail, advanced)

    def _getitem_next_regular_missing(self, head, tail, advanced, raw, length):
        # if this is in a tuple-slice and really should be 0, it will be trimmed later
        length = 1 if length == 0 else length
        nplike = self.nplike

        index = ak._v2.index.Index64(head._index)
        outindex = ak._v2.index.Index64.empty(len(index) * length, nplike)

        self._handle_error(
            nplike["awkward_missing_repeat", outindex.dtype.type, index.dtype.type](
                outindex.to(nplike),
                index.to(nplike),
                len(index),
                length,
                raw._size,
            )
        )

        out = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outindex, raw.content, None, self._parameters
        )

        return ak._v2.contents.regulararray.RegularArray(
            out._simplify_optiontype(), len(index), 1, None, self._parameters
        )

    def _getitem_next_missing_jagged(self, head, tail, advanced, that):
        nplike = self.nplike
        jagged = head.content.toListOffsetArray64()
        if not jagged:
            raise ValueError(
                "logic error: calling getitem_next_missing_jagged with bad slice type"
            )

        index = ak._v2.index.Index64(head._index)
        content = that._getitem_at(0)
        if len(content) < len(index):
            raise ValueError(
                "cannot fit masked jagged slice with length {0} into {1} of size {2}".format(
                    len(index), type(that).__name__, len(content)
                )
            )

        outputmask = ak._v2.index.Index64.empty(len(index), nplike)
        starts = ak._v2.index.Index64.empty(len(index), nplike)
        stops = ak._v2.index.Index64.empty(len(index), nplike)

        self._handle_error(
            nplike[
                "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
                index.dtype.type,
                jagged._offsets.dtype.type,
                outputmask.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
            ](
                index.to(nplike),
                jagged._offsets.to(nplike),
                outputmask.to(nplike),
                starts.to(nplike),
                stops.to(nplike),
                len(index),
            )
        )

        tmp = content._getitem_next_jagged(starts, stops, jagged.content, tail)
        out = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outputmask, tmp, None, self._parameters
        )

        return ak._v2.contents.regulararray.RegularArray(
            out._simplify_optiontype(), len(index), 1, None, self._parameters
        )

    def _getitem_next_missing(self, head, tail, advanced):
        if advanced is not None:
            raise ValueError(
                "cannot mix missing values in slice with NumPy-style advanced indexing"
            )

        if isinstance(head.content, ak._v2.contents.listoffsetarray.ListOffsetArray):
            if len(self) != 1:
                raise NotImplementedError("reached a not-well-considered code path")
            return self._getitem_next_missing_jagged(head, tail, advanced, self)

        if isinstance(head.content, ak._v2.contents.numpyarray.NumpyArray):
            headcontent = ak._v2.index.Index64(head.content.data)
            nextcontent = self._getitem_next(headcontent, tail, advanced)

        else:
            nextcontent = self._getitem_next_missing(head.content, tail, advanced)

        if isinstance(nextcontent, ak._v2.contents.regulararray.RegularArray):
            return self._getitem_next_regular_missing(
                head, tail, advanced, nextcontent, len(nextcontent)
            )

        elif isinstance(nextcontent, ak._v2.contents.recordarray.RecordArray):
            if len(nextcontent._keys) == 0:
                return nextcontent

            contents = []

            for content in nextcontent.contents:
                if isinstance(content, ak._v2.contents.regulararray.RegularArray):
                    contents.append(
                        self._getitem_next_regular_missing(
                            head, tail, advanced, content, len(content)
                        )
                    )
                else:
                    raise ValueError(
                        "FIXME: unhandled case of SliceMissing with RecordArray containing {0}".format(
                            content
                        )
                    )

            return ak._v2.contents.recordarray.RecordArray(
                contents, nextcontent._keys, None, None, self._parameters
            )

        else:
            raise ValueError(
                "FIXME: unhandled case of SliceMissing with {0}".format(nextcontent)
            )

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

                nextwhere = ak._v2._slicing.getitem_broadcast(
                    [ak._v2._slicing.prepare_tuple_item(x) for x in where],
                    self.nplike,
                )

                next = ak._v2.contents.RegularArray(self, len(self), 1, None, None)
                out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
                if len(out) == 0:
                    return out._getitem_nothing()
                else:
                    return out._getitem_at(0)

            elif isinstance(where, ak.highlevel.Array):
                return self.__getitem__(where.layout)

            elif isinstance(where, ak.layout.Content):
                return self.__getitem__(v1_to_v2(where))

            elif isinstance(where, ak._v2.contents.emptyarray.EmptyArray):
                return where.toNumpyArray(np.int64)

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
                    if len(where.data.shape) == 1:
                        where = self.nplike.nonzero(where.data)[0]
                        carry = ak._v2.index.Index64(where)
                        allow_lazy = "copied"  # True, but also can be modified in-place
                    else:
                        wheres = self.nplike.nonzero(where.data)
                        return self.__getitem__(wheres)
                else:
                    raise TypeError(
                        "array slice must be an array of integers or booleans, not\n\n    {0}".format(
                            repr(where.data).replace("\n", "\n    ")
                        )
                    )

                out = ak._v2._slicing.getitem_next_array_wrap(
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
                layout = v1_to_v2(ak.operations.convert.to_layout(where))
                as_nplike = layout.maybe_to_nplike(layout.nplike)
                if as_nplike is None:
                    return self.__getitem__(layout)
                else:
                    return self.__getitem__(ak._v2.contents.NumpyArray(as_nplike))

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
                    "axis={0} exceeds the depth of this array ({1})".format(axis, depth)
                )
            return posaxis
        elif mindepth + axis == 0:
            raise np.AxisError(
                "axis={0} exceeds the depth of this array ({1})".format(axis, depth)
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

    @property
    def keys(self):
        return self.Form.keys.__get__(self)
