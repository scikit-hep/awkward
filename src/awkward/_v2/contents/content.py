# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
import awkward._v2._reducers
from awkward._v2._slicing import NestedIndexError
from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1

np = ak.nplike.NumpyMetadata.instance()


class Content(object):
    def _init(self, identifier, parameters):
        if identifier is not None and not isinstance(
            identifier, ak._v2.identifier.Identifier
        ):
            raise TypeError(
                "{0} 'identifier' must be an Identifier or None, not {1}".format(
                    type(self).__name__, repr(identifier)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be a dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )

        self._identifier = identifier
        self._parameters = parameters

    @property
    def identifier(self):
        return self._identifier

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = {}
        return self._parameters

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    def _repr_extra(self, indent):
        out = []
        if self._parameters is not None:
            for k, v in self._parameters.items():
                out.append(
                    "\n{0}<parameter name={1}>{2}</parameter>".format(
                        indent, repr(k), repr(v)
                    )
                )
        if self._identifier is not None:
            out.append(self._identifier._repr("\n" + indent, "", ""))
        return out

    def _simplify_optiontype(self):
        return self

    def _simplify_uniontype(self):
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
        nplike = head.nplike

        index = ak._v2.index.Index64(head.index)
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
        nplike = head.nplike
        jagged = head.content.toListOffsetArray64()

        index = ak._v2.index.Index64(head._index)
        content = that._getitem_at(0)
        if len(content) < len(index) and nplike.known_shape:
            raise NestedIndexError(
                self,
                head,
                "cannot fit masked jagged slice with length {0} into {1} of size {2}".format(
                    len(index), type(that).__name__, len(content)
                ),
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
        assert isinstance(head, ak._v2.contents.IndexedOptionArray)

        if advanced is not None:
            raise NestedIndexError(
                self,
                head,
                "cannot mix missing values in slice with NumPy-style advanced indexing",
            )

        if isinstance(head.content, ak._v2.contents.listoffsetarray.ListOffsetArray):
            if len(self) != 1:
                raise NotImplementedError("reached a not-well-considered code path")
            return self._getitem_next_missing_jagged(head, tail, advanced, self)

        if isinstance(head.content, ak._v2.contents.numpyarray.NumpyArray):
            headcontent = ak._v2.index.Index64(head.content.data)
            nextcontent = self._getitem_next(headcontent, tail, advanced)
        else:
            nextcontent = self._getitem_next(head.content, tail, advanced)

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
                    raise NotImplementedError(
                        "FIXME: unhandled case of SliceMissing with RecordArray containing {0}".format(
                            content
                        )
                    )

            return ak._v2.contents.recordarray.RecordArray(
                contents, nextcontent._keys, None, None, self._parameters
            )

        else:
            raise NotImplementedError(
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

                items = [ak._v2._slicing.prepare_tuple_item(x) for x in where]
                nextwhere = ak._v2._slicing.getitem_broadcast(
                    items, ak.nplike.of(*items)
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

            elif (
                isinstance(where, Content)
                and where._parameters is not None
                and (where._parameters.get("__array__") in ("string", "bytestring"))
            ):
                return self._getitem_fields(ak.to_list(where))

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
                        where = where.nplike.nonzero(where.data)[0]
                        carry = ak._v2.index.Index64(where)
                        allow_lazy = "copied"  # True, but also can be modified in-place
                    else:
                        wheres = where.nplike.nonzero(where.data)
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

            try:
                tmp = "    " + repr(ak.Array(v2_to_v1(self)))
            except Exception:
                tmp = self._repr("    ", "", "")
            raise IndexError(
                """cannot slice

{0}

with

    {1}

at inner {2} of length {3}, using sub-slice {4}.{5}""".format(
                    tmp,
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

    def _typetracer_identifier(self):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

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
        mindepth, maxdepth = self.minmax_depth
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

    def _reduce(self, reducer, axis=-1, mask=True, keepdims=False):
        if axis is None:
            raise NotImplementedError

        negaxis = -axis
        branch, depth = self.branch_depth

        if branch:
            if negaxis <= 0:
                raise ValueError(
                    "cannot use non-negative axis on a nested list structure "
                    "of variable depth (negative axis counts from the leaves of "
                    "the tree; non-negative from the root)"
                )
            if negaxis > depth:
                raise ValueError(
                    "cannot use axis={0} on a nested list structure that splits into "
                    "different depths, the minimum of which is depth={1} "
                    "from the leaves".format(axis, depth)
                )
        else:
            if negaxis <= 0:
                negaxis += depth
            if not (0 < negaxis and negaxis <= depth):
                raise ValueError(
                    "axis={0} exceeds the depth of the nested list structure "
                    "(which is {1})".format(axis, depth)
                )

        parents = ak._v2.index.Index64.zeros(len(self), self.nplike)
        starts = ak._v2.index.Index64.zeros(1, self.nplike)
        shifts = None
        next = self._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            1,
            mask,
            keepdims,
        )

        return next[0]

    def argmin(self, axis=-1, mask=True, keepdims=False):
        return self._reduce(awkward._v2._reducers.ArgMin, axis, mask, keepdims)

    def argmax(self, axis=-1, mask=True, keepdims=False):
        return self._reduce(awkward._v2._reducers.ArgMax, axis, mask, keepdims)

    def count(self, axis=-1, mask=False, keepdims=False):
        return self._reduce(awkward._v2._reducers.Count, axis, mask, keepdims)

    def count_nonzero(self, axis=-1, mask=False, keepdims=False):
        return self._reduce(awkward._v2._reducers.CountNonzero, axis, mask, keepdims)

    def sum(self, axis=-1, mask=False, keepdims=False):
        return self._reduce(awkward._v2._reducers.Sum, axis, mask, keepdims)

    def prod(self, axis=-1, mask=False, keepdims=False):
        return self._reduce(awkward._v2._reducers.Prod, axis, mask, keepdims)

    def any(self, axis=-1, mask=False, keepdims=False):
        return self._reduce(awkward._v2._reducers.Any, axis, mask, keepdims)

    def all(self, axis=-1, mask=False, keepdims=False):
        return self._reduce(awkward._v2._reducers.All, axis, mask, keepdims)

    def min(self, axis=-1, mask=True, keepdims=False, initial=None):
        return self._reduce(awkward._v2._reducers.Min(initial), axis, mask, keepdims)

    def max(self, axis=-1, mask=True, keepdims=False, initial=None):
        return self._reduce(awkward._v2._reducers.Max(initial), axis, mask, keepdims)

    def argsort(self, axis=-1, ascending=True, stable=False, kind=None, order=None):
        negaxis = -axis
        branch, depth = self.branch_depth
        if branch:
            if negaxis <= 0:
                raise ValueError(
                    "cannot use non-negative axis on a nested list structure "
                    "of variable depth (negative axis counts from the leaves "
                    "of the tree; non-negative from the root)"
                )
            if negaxis > depth:
                raise ValueError(
                    "cannot use axis={0} on a nested list structure that splits into "
                    "different depths, the minimum of which is depth={1} from the leaves".format(
                        axis, depth
                    )
                )
        else:
            if negaxis <= 0:
                negaxis = negaxis + depth
            if not (0 < negaxis and negaxis <= depth):
                raise ValueError(
                    "axis={0} exceeds the depth of the nested list structure "
                    "(which is {1})".format(axis, depth)
                )

        starts = ak._v2.index.Index64.zeros(1, self.nplike)
        parents = ak._v2.index.Index64.zeros(len(self), self.nplike)
        return self._argsort_next(
            negaxis,
            starts,
            None,
            parents,
            1,
            ascending,
            stable,
            kind,
            order,
        )

    def sort(self, axis=-1, ascending=True, stable=False, kind=None, order=None):
        negaxis = -axis
        branch, depth = self.branch_depth
        if branch:
            if negaxis <= 0:
                raise ValueError(
                    "cannot use non-negative axis on a nested list structure "
                    "of variable depth (negative axis counts from the leaves "
                    "of the tree; non-negative from the root)"
                )
            if negaxis > depth:
                raise ValueError(
                    "cannot use axis={0} on a nested list structure that splits into "
                    "different depths, the minimum of which is depth={1} from the leaves".format(
                        axis, depth
                    )
                )
        else:
            if negaxis <= 0:
                negaxis = negaxis + depth
            if not (0 < negaxis and negaxis <= depth):
                raise ValueError(
                    "axis={0} exceeds the depth of the nested list structure "
                    "(which is {1})".format(axis, depth)
                )

        starts = ak._v2.index.Index64.zeros(1, self.nplike)
        parents = ak._v2.index.Index64.zeros(len(self), self.nplike)
        return self._sort_next(
            negaxis,
            starts,
            parents,
            1,
            ascending,
            stable,
            kind,
            order,
        )

    def _combinations_axis0(self, n, replacement, recordlookup, parameters):
        size = len(self)
        if replacement:
            size = size + (n - 1)
        thisn = n
        combinationslen = 0
        if thisn > size:
            combinationslen = 0
        elif thisn == size:
            combinationslen = 1
        else:
            if thisn * 2 > size:
                thisn = size - thisn
            combinationslen = size
            for j in range(2, thisn + 1):
                combinationslen = combinationslen * (size - j + 1)
                combinationslen = combinationslen // j

        nplike = self.nplike
        tocarryraw = nplike.empty(n, dtype=np.intp)
        tocarry = []
        for i in range(n):
            ptr = ak._v2.index.Index64.empty(combinationslen, nplike, dtype=np.int64)
            tocarry.append(ptr)
            tocarryraw[i] = ptr.ptr

        toindex = ak._v2.index.Index64.empty(n, nplike, dtype=np.int64)
        fromindex = ak._v2.index.Index64.empty(n, nplike, dtype=np.int64)

        self._handle_error(
            nplike[
                "awkward_RegularArray_combinations_64",
                np.int64,
                toindex.to(nplike).dtype.type,
                fromindex.to(nplike).dtype.type,
            ](
                tocarryraw,
                toindex.to(nplike),
                fromindex.to(nplike),
                n,
                replacement,
                len(self),
                1,
            )
        )
        contents = []
        for ptr in tocarry:
            contents.append(ak._v2.contents.IndexedArray(ptr, self))
        return ak._v2.contents.recordarray.RecordArray(
            contents, recordlookup, parameters=parameters
        )

    def combinations(self, n, replacement=False, axis=1, keys=None, parameters=None):
        if n < 1:
            raise ValueError("in combinations, 'n' must be at least 1")

        recordlookup = None
        if keys is not None:
            recordlookup = keys
            if len(recordlookup) != n:
                raise ValueError("if provided, the length of 'keys' must be 'n'")
        return self._combinations(n, replacement, recordlookup, parameters, axis, 0)

    def validityerror_parameters(self, path):
        if self.parameter("__array__") == "string":
            content = None
            if isinstance(
                self,
                (
                    ak._v2.contents.listarray.ListArray,
                    ak._v2.contents.listoffsetarray.ListOffsetArray,
                    ak._v2.contents.regulararray.RegularArray,
                ),
            ):
                content = self.content
            else:
                return 'at {0} ("{1}"): __array__ = "string" only allowed for ListArray, ListOffsetArray and RegularArray'.format(
                    path, type(self)
                )
            if content.parameter("__array__") != "char":
                return 'at {0} ("{1}"): __array__ = "string" must directly contain a node with __array__ = "char"'.format(
                    path, type(self)
                )
            if isinstance(content, ak._v2.contents.numpyarray.NumpyArray):
                if content.dtype == np.uint8:
                    return 'at {0} ("{1}"): __array__ = "char" requires dtype == uint8'.format(
                        path, type(self)
                    )
                if len(content.shape) != 1:
                    return 'at {0} ("{1}"): __array__ = "char" must be one-dimensional'.format(
                        path, type(self)
                    )
            else:
                return 'at {0} ("{1}"): __array__ = "char" only allowed for NumpyArray'.format(
                    path, type(self)
                )
            return ""

        if self.parameter("__array__") == "bytestring":
            content = None
            if isinstance(
                self,
                (
                    ak._v2.contents.listarray.ListArray,
                    ak._v2.contents.listoffsetarray.ListOffsetArray,
                    ak._v2.contents.regulararray.RegularArray,
                ),
            ):
                content = self.content
            else:
                return 'at {0} ("{1}"): __array__ = "bytestring" only allowed for ListArray, ListOffsetArray and RegularArray'.format(
                    path, type(self)
                )
            if content.parameter("__array__") != "byte":
                return 'at {0} ("{1}"): __array__ = "bytestring" must directly contain a node with __array__ = "byte"'.format(
                    path, type(self)
                )
            if isinstance(content, ak._v2.contents.numpyarray.NumpyArray):
                if content.dtype == np.uint8:
                    return 'at {0} ("{1}"): __array__ = "byte" requires dtype == uint8'.format(
                        path, type(self)
                    )
                if len(content.shape) != 1:
                    return 'at {0} ("{1}"): __array__ = "byte" must be one-dimensional'.format(
                        path, type(self)
                    )
            else:
                return 'at {0} ("{1}"): __array__ = "byte" only allowed for NumpyArray'.format(
                    path, type(self)
                )
            return ""

        if self.parameter("__array__") == "char":
            return 'at {0} ("{1}"): __array__ = "char" must be directly inside __array__ = "string"'.format(
                path, type(self)
            )

        if self.parameter("__array__") == "byte":
            return 'at {0} ("{1}"): __array__ = "byte" must be directly inside __array__ = "bytestring"'.format(
                path, type(self)
            )

        if self.parameter("__array__") == "categorical":
            content = None
            if isinstance(
                self,
                (
                    ak._v2.contents.indexedarray.IndexedArray,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ),
            ):
                content = self.content
            else:
                return 'at {0} ("{1}"): __array__ = "string" only allowed for IndexedArray and IndexedOptionArray'.format(
                    path, type(self)
                )
            return NotImplementedError("TODO: Implement is_unique")

        return ""

    def validityerror(self, path="layout"):
        paramcheck = self.validityerror_parameters(path)
        if paramcheck != "":
            return paramcheck
        return self._validityerror(path)

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

    @property
    def dimension_optiontype(self):
        return self.Form.dimension_optiontype.__get__(self)
