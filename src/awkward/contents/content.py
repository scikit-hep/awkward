# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sized
from numbers import Complex, Integral, Real
from typing import Any, TypeVar

import awkward as ak
import awkward._reducers
from awkward.forms import form
from awkward.nplikes import NumpyLike
from awkward.typing import Self, TypeAlias

np = ak.nplikes.NumpyMetadata.instance()
numpy = ak.nplikes.Numpy.instance()

AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)
ActionType: TypeAlias = """Callable[
    [
        Content,
        int,
        dict | None,
        dict | None,
        Callable[[], None],
        dict | None,
        NumpyLike | None,
        dict[str, Any],
    ],
    Content | None,
]"""


# FIXME: introduce sentinel type for this
class _Unset:
    def __repr__(self):
        return f"{__name__}.unset"


unset = _Unset()


class Content:
    is_numpy = False
    is_unknown = False
    is_list = False
    is_regular = False
    is_option = False
    is_indexed = False
    is_record = False
    is_union = False

    def _init(self, parameters: dict[str, Any] | None, nplike: NumpyLike | None):
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'parameters' must be a dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )

        if nplike is not None and not isinstance(nplike, NumpyLike):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'nplike' must be an NumpyLike or None, not {}".format(
                        type(self).__name__, repr(nplike)
                    )
                )
            )

        self._parameters = parameters
        self._nplike = nplike

    @property
    def parameters(self) -> dict[str, Any]:
        if self._parameters is None:
            self._parameters = {}
        return self._parameters

    def parameter(self, key: str):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    @property
    def nplike(self) -> NumpyLike | None:
        return self._nplike

    @property
    def form(self) -> form.Form:
        return self.form_with_key(None)

    def form_with_key(self, form_key="node{id}", id_start=0):
        hold_id = [id_start]

        if form_key is None:

            def getkey(layout):
                return None

        elif isinstance(form_key, str):

            def getkey(layout):
                out = form_key.format(id=hold_id[0])
                hold_id[0] += 1
                return out

        elif callable(form_key):

            def getkey(layout):
                out = form_key(id=hold_id[0], layout=layout)
                hold_id[0] += 1
                return out

        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "form_key must be None, a string, or a callable, not {}".format(
                        type(form_key)
                    )
                )
            )

        return self._form_with_key(getkey)

    def _form_with_key(
        self,
        getkey: Callable[[Content], form.Form],
    ) -> form.Form:
        raise ak._util.error(NotImplementedError)

    @property
    def Form(self) -> type[form.Form]:
        raise ak._util.error(NotImplementedError)

    @property
    def typetracer(self) -> Self:
        raise ak._util.error(NotImplementedError)

    @property
    def length(self) -> int:
        raise ak._util.error(NotImplementedError)

    def forget_length(self) -> Self:
        if not isinstance(self._nplike, ak._typetracer.TypeTracer):
            return self.typetracer._forget_length()
        else:
            return self._forget_length()

    def _forget_length(self) -> Self:
        raise ak._util.error(NotImplementedError)

    def to_buffers(
        self,
        container: MutableMapping[str, Any] | None = None,
        buffer_key="{form_key}-{attribute}",
        form_key: str | None = "node{id}",
        id_start: Integral = 0,
        nplike: NumpyLike | None = None,
    ) -> tuple[form.Form, int, Mapping[str, Any]]:
        if container is None:
            container = {}
        if nplike is None:
            nplike = self._nplike
        if not nplike.known_data:
            raise ak._errors.wrap_error(
                TypeError("cannot call 'to_buffers' on an array without concrete data")
            )

        if isinstance(buffer_key, str):

            def getkey(layout, form, attribute):
                return buffer_key.format(form_key=form.form_key, attribute=attribute)

        elif callable(buffer_key):

            def getkey(layout, form, attribute):
                return buffer_key(
                    form_key=form.form_key,
                    attribute=attribute,
                    layout=layout,
                    form=form,
                )

        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "buffer_key must be a string or a callable, not {}".format(
                        type(buffer_key)
                    )
                )
            )

        if form_key is None:
            raise ak._errors.wrap_error(
                TypeError(
                    "a 'form_key' must be supplied, to match Form elements to buffers in the 'container'"
                )
            )

        form = self.form_with_key(form_key=form_key, id_start=id_start)

        self._to_buffers(form, getkey, container, nplike)

        return form, len(self), container

    def _to_buffers(
        self,
        form: form.Form,
        getkey: Callable[[Content, form.Form, str], str],
        container: MutableMapping[str, Any] | None,
        nplike: NumpyLike | None,
    ) -> tuple[form.Form, int, Mapping[str, Any]]:
        raise ak._util.error(NotImplementedError)

    def __len__(self) -> int:
        return self.length

    def _repr_extra(self, indent: str) -> list[str]:
        out = []
        if self._parameters is not None:
            for k, v in self._parameters.items():
                out.append(
                    "\n{}<parameter name={}>{}</parameter>".format(
                        indent, repr(k), repr(v)
                    )
                )
        return out

    def maybe_to_array(self, nplike):
        return None

    def _handle_error(self, error, slicer=None):
        if error is not None and error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")

            message = error.str.decode(errors="surrogateescape")

            if error.pass_through:
                raise ak._errors.wrap_error(ValueError(message + filename))

            else:
                if error.attempt != ak._util.kSliceNone:
                    message += f" while attempting to get index {error.attempt}"

                message += filename

                if slicer is None:
                    raise ak._errors.wrap_error(ValueError(message))
                else:
                    raise ak._errors.index_error(self, slicer, message)

    @staticmethod
    def _selfless_handle_error(error):
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")

            message = error.str.decode(errors="surrogateescape")

            if error.pass_through:
                raise ak._errors.wrap_error(ValueError(message + filename))

            else:
                if error.attempt != ak._util.kSliceNone:
                    message += f" while attempting to get index {error.attempt}"

                message += filename

                raise ak._errors.wrap_error(ValueError(message))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise ak._errors.wrap_error(
            TypeError(
                "do not apply NumPy functions to low-level layouts (Content subclasses); put them in ak.highlevel.Array"
            )
        )

    def __array_function__(self, func, types, args, kwargs):
        raise ak._errors.wrap_error(
            TypeError(
                "do not apply NumPy functions to low-level layouts (Content subclasses); put them in ak.highlevel.Array"
            )
        )

    def __array__(self, **kwargs):
        raise ak._errors.wrap_error(
            TypeError(
                "do not try to convert low-level layouts (Content subclasses) into NumPy arrays; put them in ak.highlevel.Array"
            )
        )

    def __iter__(self):
        if not self._nplike.known_data:
            raise ak._errors.wrap_error(
                TypeError("cannot iterate on an array without concrete data")
            )

        for i in range(len(self)):
            yield self._getitem_at(i)

    def _getitem_next_field(self, head, tail, advanced: ak.index.Index | None):
        nexthead, nexttail = ak._slicing.headtail(tail)
        return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

    def _getitem_next_fields(self, head, tail, advanced: ak.index.Index | None):
        only_fields, not_fields = [], []
        for x in tail:
            if isinstance(x, (str, list)):
                only_fields.append(x)
            else:
                not_fields.append(x)
        nexthead, nexttail = ak._slicing.headtail(tuple(not_fields))
        return self._getitem_fields(head, tuple(only_fields))._getitem_next(
            nexthead, nexttail, advanced
        )

    def _getitem_next_newaxis(self, tail, advanced: ak.index.Index | None):
        nexthead, nexttail = ak._slicing.headtail(tail)
        return ak.contents.RegularArray(
            self._getitem_next(nexthead, nexttail, advanced),
            1,  # size
            0,  # zeros_length is irrelevant when the size is 1 (!= 0)
            None,
            self._nplike,
        )

    def _getitem_next_ellipsis(self, tail, advanced: ak.index.Index | None):
        mindepth, maxdepth = self.minmax_depth

        dimlength = sum(
            1 if isinstance(x, (int, slice, ak.index.Index64)) else 0 for x in tail
        )

        if len(tail) == 0 or mindepth - 1 == maxdepth - 1 == dimlength:
            nexthead, nexttail = ak._slicing.headtail(tail)
            return self._getitem_next(nexthead, nexttail, advanced)

        elif dimlength in {mindepth - 1, maxdepth - 1}:
            raise ak._errors.index_error(
                self,
                Ellipsis,
                "ellipsis (`...`) can't be used on data with different numbers of dimensions",
            )

        else:
            return self._getitem_next(slice(None), (Ellipsis,) + tail, advanced)

    def _getitem_next_regular_missing(
        self, head, tail, advanced: ak.index.Index | None, raw, length: Integral
    ):
        # if this is in a tuple-slice and really should be 0, it will be trimmed later
        length = 1 if length == 0 else length
        index = ak.index.Index64(head.index, nplike=self.nplike)
        indexlength = index.length
        index = index._to_nplike(self.nplike)
        outindex = ak.index.Index64.empty(index.length * length, self._nplike)

        assert outindex.nplike is self._nplike and index.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_missing_repeat", outindex.dtype.type, index.dtype.type
            ](
                outindex.data,
                index.data,
                index.length,
                length,
                raw._size,
            ),
            slicer=head,
        )

        out = ak.contents.IndexedOptionArray(
            outindex, raw.content, self._parameters, self._nplike
        )

        return ak.contents.RegularArray(
            out.simplify_optiontype(),
            indexlength,
            1,
            self._parameters,
            self._nplike,
        )

    def _getitem_next_missing_jagged(
        self, head, tail, advanced: ak.index.Index | None, that
    ):
        head = head._to_nplike(self._nplike)
        jagged = head.content.toListOffsetArray64()

        index = ak.index.Index64(head._index, nplike=self.nplike)
        content = that._getitem_at(0)
        if self._nplike.known_shape and content.length < index.length:
            raise ak._errors.index_error(
                self,
                head,
                "cannot fit masked jagged slice with length {} into {} of size {}".format(
                    index.length, type(that).__name__, content.length
                ),
            )

        outputmask = ak.index.Index64.empty(index.length, self._nplike)
        starts = ak.index.Index64.empty(index.length, self._nplike)
        stops = ak.index.Index64.empty(index.length, self._nplike)

        assert (
            index.nplike is self._nplike
            and jagged._offsets.nplike is self._nplike
            and outputmask.nplike is self._nplike
            and starts.nplike is self._nplike
            and stops.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
                "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
                index.dtype.type,
                jagged._offsets.dtype.type,
                outputmask.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
            ](
                index.data,
                jagged._offsets.data,
                outputmask.data,
                starts.data,
                stops.data,
                index.length,
            ),
            slicer=head,
        )

        tmp = content._getitem_next_jagged(starts, stops, jagged.content, tail)
        out = ak.contents.IndexedOptionArray(
            outputmask, tmp, self._parameters, self._nplike
        )
        return ak.contents.RegularArray(
            out.simplify_optiontype(),
            index.length,
            1,
            self._parameters,
            self._nplike,
        )

    def _getitem_next_missing(
        self,
        head: ak.contents.IndexedOptionArray,
        tail,
        advanced: ak.index.Index | None,
    ):
        assert isinstance(head, ak.contents.IndexedOptionArray)

        if advanced is not None:
            raise ak._errors.index_error(
                self,
                head,
                "cannot mix missing values in slice with NumPy-style advanced indexing",
            )

        if isinstance(head.content, ak.contents.ListOffsetArray):
            if self.nplike.known_shape and self.length != 1:
                raise ak._errors.wrap_error(
                    NotImplementedError("reached a not-well-considered code path")
                )
            return self._getitem_next_missing_jagged(head, tail, advanced, self)

        if isinstance(head.content, ak.contents.NumpyArray):
            headcontent = ak.index.Index64(head.content.data)
            nextcontent = self._getitem_next(headcontent, tail, advanced)
        else:
            nextcontent = self._getitem_next(head.content, tail, advanced)

        if isinstance(nextcontent, ak.contents.RegularArray):
            return self._getitem_next_regular_missing(
                head, tail, advanced, nextcontent, nextcontent.length
            )

        elif isinstance(nextcontent, ak.contents.RecordArray):
            if len(nextcontent._fields) == 0:
                return nextcontent

            contents = []

            for content in nextcontent.contents:
                if isinstance(content, ak.contents.RegularArray):
                    contents.append(
                        self._getitem_next_regular_missing(
                            head, tail, advanced, content, content.length
                        )
                    )
                else:
                    raise ak._errors.wrap_error(
                        NotImplementedError(
                            "FIXME: unhandled case of SliceMissing with RecordArray containing {}".format(
                                content
                            )
                        )
                    )

            return ak.contents.RecordArray(
                contents,
                nextcontent._fields,
                None,
                self._parameters,
                self._nplike,
            )

        else:
            raise ak._errors.wrap_error(
                NotImplementedError(
                    f"FIXME: unhandled case of SliceMissing with {nextcontent}"
                )
            )

    def __getitem__(self, where):
        return self._getitem(where)

    def _getitem(self, where):
        if ak._util.is_integer(where):
            return self._getitem_at(where)

        elif isinstance(where, slice) and where.step is None:
            return self._getitem_range(where)

        elif isinstance(where, slice):
            return self._getitem((where,))

        elif isinstance(where, str):
            return self._getitem_field(where)

        elif where is np.newaxis:
            return self._getitem((where,))

        elif where is Ellipsis:
            return self._getitem((where,))

        elif isinstance(where, tuple):
            if len(where) == 0:
                return self

            # Normalise valid indices onto well-defined basis
            items = ak._slicing.normalise_items(where, self._nplike)
            # Prepare items for advanced indexing (e.g. via broadcasting)
            nextwhere = ak._slicing.prepare_advanced_indexing(items)

            next = ak.contents.RegularArray(
                self,
                self.length if self._nplike.known_shape else 1,
                1,
                None,
                self._nplike,
            )

            out = next._getitem_next(nextwhere[0], nextwhere[1:], None)

            if out.length == 0:
                return out._getitem_nothing()
            else:
                return out._getitem_at(0)

        elif isinstance(where, ak.highlevel.Array):
            return self._getitem(where.layout)

        elif isinstance(where, ak.highlevel.Array):
            return self._getitem(where.layout)

        elif (
            isinstance(where, Content)
            and where._parameters is not None
            and (where._parameters.get("__array__") in ("string", "bytestring"))
        ):
            return self._getitem_fields(ak.operations.to_list(where))

        elif isinstance(where, ak.contents.EmptyArray):
            return where.toNumpyArray(np.int64)

        elif isinstance(where, ak.contents.NumpyArray):
            if issubclass(where.dtype.type, np.int64):
                carry = ak.index.Index64(where.data.reshape(-1))
                allow_lazy = True
            elif issubclass(where.dtype.type, np.integer):
                carry = ak.index.Index64(
                    where.data.astype(np.int64).reshape(-1), nplike=self.nplike
                )
                allow_lazy = "copied"  # True, but also can be modified in-place
            elif issubclass(where.dtype.type, (np.bool_, bool)):
                if len(where.data.shape) == 1:
                    where = self._nplike.nonzero(where.data)[0]
                    carry = ak.index.Index64(where, nplike=self.nplike)
                    allow_lazy = "copied"  # True, but also can be modified in-place
                else:
                    wheres = self._nplike.nonzero(where.data)
                    return self._getitem(wheres)
            else:
                raise ak._errors.wrap_error(
                    TypeError(
                        "array slice must be an array of integers or booleans, not\n\n    {}".format(
                            repr(where.data).replace("\n", "\n    ")
                        )
                    )
                )

            out = ak._slicing.getitem_next_array_wrap(
                self._carry(carry, allow_lazy), where.shape
            )
            if out.length == 0:
                return out._getitem_nothing()
            else:
                return out._getitem_at(0)

        elif isinstance(where, Content):
            return self._getitem((where,))

        elif ak._util.is_sized_iterable(where) and len(where) == 0:
            return self._carry(ak.index.Index64.empty(0, self._nplike), allow_lazy=True)

        elif ak._util.is_sized_iterable(where) and all(
            isinstance(x, str) for x in where
        ):
            return self._getitem_fields(where)

        elif ak._util.is_sized_iterable(where):
            layout = ak.operations.to_layout(where)
            as_array = layout.maybe_to_array(layout.nplike)
            if as_array is None:
                return self._getitem(layout)
            else:
                return self._getitem(
                    ak.contents.NumpyArray(as_array, None, layout.nplike)
                )

        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                    "integer/boolean arrays (possibly with variable-length nested "
                    "lists or missing values), field name (str) or names (non-tuple "
                    "iterable of str) are valid indices for slicing, not\n\n    "
                    + repr(where).replace("\n", "\n    ")
                )
            )

    def _getitem_at(self, where: Integral):
        raise ak._util.error(NotImplementedError)

    def _getitem_range(self, where: slice):
        raise ak._util.error(NotImplementedError)

    def _getitem_field(self, where: str):
        raise ak._util.error(NotImplementedError)

    def _getitem_fields(self, where: list[str], only_fields: tuple[str, ...] = ()):
        raise ak._util.error(NotImplementedError)

    def _getitem_next(self, head, tail, advanced: ak.index.Index | None):
        raise ak._util.error(NotImplementedError)

    def _carry(self, carry: ak.index.Index, allow_lazy: bool):
        raise ak._util.error(NotImplementedError)

    def _carry_asrange(self, carry: ak.index.Index):
        assert isinstance(carry, ak.index.Index)

        result = self._nplike.index_nplike.empty(1, dtype=np.bool_)
        assert carry.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_Index_iscontiguous",  # badly named
                np.bool_,
                carry.dtype.type,
            ](
                result,
                carry.data,
                carry.length,
            ),
            slicer=carry.data,
        )
        if result[0]:
            if carry.length == self.length:
                return self
            elif carry.length < self.length:
                return self._getitem_range(slice(0, carry.length))
            else:
                raise ak._errors.wrap_error(IndexError)
        else:
            return None

    def axis_wrap_if_negative(self, axis: AxisMaybeNone) -> AxisMaybeNone:
        if axis is None or axis >= 0:
            return axis

        mindepth, maxdepth = self.minmax_depth
        depth = self.purelist_depth
        if mindepth == depth and maxdepth == depth:
            posaxis = depth + axis
            if posaxis < 0:
                raise ak._errors.wrap_error(
                    np.AxisError(
                        f"axis={axis} exceeds the depth ({depth}) of this array"
                    )
                )
            return posaxis

        elif mindepth + axis == 0:
            raise ak._errors.wrap_error(
                np.AxisError(
                    "axis={} exceeds the depth ({}) of at least one record field (or union possibility) of this array".format(
                        axis, depth
                    )
                )
            )

        return axis

    def _local_index_axis0(self) -> ak.contents.NumpyArray:
        localindex = ak.index.Index64.empty(self.length, self._nplike)
        self._handle_error(
            self._nplike["awkward_localindex", np.int64](
                localindex.data,
                localindex.length,
            )
        )
        return ak.contents.NumpyArray(localindex, None, self._nplike)

    def merge(self, other: Content) -> Content:
        others = [other]
        return self.mergemany(others)

    def mergeable(self, other: Content, mergebool: bool = True) -> bool:
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_union:
            return True
        # Otherwise, do the parameters match? If not, we can't merge.
        elif not (
            form._parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            )
        ):
            return False
        # Finally, fall back upon the per-content implementation
        else:
            return self._mergeable(other, mergebool)

    def _mergeable(self, other: Content, mergebool: bool) -> bool:
        raise ak._errors.wrap_error(NotImplementedError)

    def mergemany(self, others: list[Content]) -> Content:
        raise ak._errors.wrap_error(NotImplementedError)

    def merge_as_union(self, other: Content) -> ak.contents.UnionArray:
        mylength = self.length
        theirlength = other.length
        tags = ak.index.Index8.empty((mylength + theirlength), self._nplike)
        index = ak.index.Index64.empty((mylength + theirlength), self._nplike)
        contents = [self, other]
        assert tags.nplike is self._nplike
        self._handle_error(
            self._nplike["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data, 0, mylength, 0
            )
        )
        assert index.nplike is self._nplike
        self._handle_error(
            self._nplike["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data, 0, mylength
            )
        )
        self._handle_error(
            self._nplike["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data, mylength, theirlength, 1
            )
        )
        self._handle_error(
            self._nplike["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data, mylength, theirlength
            )
        )

        return ak.contents.UnionArray(tags, index, contents, None, self._nplike)

    def _merging_strategy(
        self, others: list[Content]
    ) -> tuple[list[Content], list[Content]]:
        if len(others) == 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "to merge this array with 'others', at least one other must be provided"
                )
            )

        head = [self]
        tail = []
        i = 0

        while i < len(others):
            other = others[i]
            if isinstance(
                other,
                (
                    ak.contents.IndexedArray,
                    ak.contents.IndexedOptionArray,
                    ak.contents.ByteMaskedArray,
                    ak.contents.BitMaskedArray,
                    ak.contents.UnmaskedArray,
                    ak.contents.UnionArray,
                ),
            ):
                break
            else:
                head.append(other)
            i = i + 1

        while i < len(others):
            tail.append(others[i])
            i = i + 1

        if any(isinstance(x.nplike, ak._typetracer.TypeTracer) for x in head + tail):
            head = [
                x if isinstance(x.nplike, ak._typetracer.TypeTracer) else x.typetracer
                for x in head
            ]
            tail = [
                x if isinstance(x.nplike, ak._typetracer.TypeTracer) else x.typetracer
                for x in tail
            ]

        return (head, tail)

    def dummy(self):
        raise ak._errors.wrap_error(
            NotImplementedError(
                "FIXME: need to implement 'dummy', which makes an array of length 1 and an arbitrary value (0?) for this array type"
            )
        )

    def local_index(self, axis: Integral):
        return self._local_index(axis, 0)

    def _local_index(self, axis: Integral, depth: Integral):
        raise ak._util.error(NotImplementedError)

    def _reduce(
        self,
        reducer: type[ak._reducers.Reducer],
        axis: Integral = -1,
        mask: bool = True,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        if axis is None:
            raise ak._errors.wrap_error(NotImplementedError)

        negaxis = -axis
        branch, depth = self.branch_depth

        if branch:
            if negaxis <= 0:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves of "
                        "the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use axis={} on a nested list structure that splits into "
                        "different depths, the minimum of which is depth={} "
                        "from the leaves".format(axis, depth)
                    )
                )
        else:
            if negaxis <= 0:
                negaxis += depth
            if not (0 < negaxis and negaxis <= depth):
                raise ak._errors.wrap_error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
                )

        starts = ak.index.Index64.zeros(1, self._nplike)
        parents = ak.index.Index64.zeros(self.length, self._nplike)
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
            behavior,
        )

        return next[0]

    def _reduce_next(
        self,
        reducer: type[ak._reducers.Reducer],
        negaxis: Integral,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        parents: ak.index.Index,
        outlength: Integral,
        mask: bool,
        keepdims: bool,
        behavior: dict | None,
    ):
        raise ak._util.error(NotImplementedError)

    def argmin(
        self,
        axis: Integral = -1,
        mask: bool = True,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.ArgMin, axis, mask, keepdims, behavior)

    def argmax(
        self,
        axis: Integral = -1,
        mask: bool = True,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.ArgMax, axis, mask, keepdims, behavior)

    def count(
        self,
        axis: Integral = -1,
        mask: bool = False,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.Count, axis, mask, keepdims, behavior)

    def count_nonzero(
        self,
        axis: Integral = -1,
        mask: bool = False,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(
            awkward._reducers.CountNonzero, axis, mask, keepdims, behavior
        )

    def sum(
        self,
        axis: Integral = -1,
        mask: bool = False,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.Sum, axis, mask, keepdims, behavior)

    def prod(
        self,
        axis: Integral = -1,
        mask: bool = False,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.Prod, axis, mask, keepdims, behavior)

    def any(
        self,
        axis: Integral = -1,
        mask: bool = False,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.Any, axis, mask, keepdims, behavior)

    def all(
        self,
        axis: Integral = -1,
        mask: bool = False,
        keepdims: bool = False,
        behavior: dict | None = None,
    ):
        return self._reduce(awkward._reducers.All, axis, mask, keepdims, behavior)

    def min(
        self,
        axis: Integral = -1,
        mask: bool = True,
        keepdims: bool = False,
        initial: dict | None = None,
        behavior=None,
    ):
        return self._reduce(
            awkward._reducers.Min(initial), axis, mask, keepdims, behavior
        )

    def max(
        self,
        axis: Integral = -1,
        mask: bool = True,
        keepdims: bool = False,
        initial: dict | None = None,
        behavior: dict = None,
    ):
        return self._reduce(
            awkward._reducers.Max(initial), axis, mask, keepdims, behavior
        )

    def argsort(
        self,
        axis: Integral = -1,
        ascending: bool = True,
        stable: bool = False,
        kind: Any = None,
        order: Any = None,
    ):
        negaxis = -axis
        branch, depth = self.branch_depth
        if branch:
            if negaxis <= 0:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves "
                        "of the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use axis={} on a nested list structure that splits into "
                        "different depths, the minimum of which is depth={} from the leaves".format(
                            axis, depth
                        )
                    )
                )
        else:
            if negaxis <= 0:
                negaxis = negaxis + depth
            if not (0 < negaxis and negaxis <= depth):
                raise ak._errors.wrap_error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
                )

        starts = ak.index.Index64.zeros(1, self._nplike)
        parents = ak.index.Index64.zeros(self.length, self._nplike)
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

    def _argsort_next(
        self,
        negaxis: Integral,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        parents: ak.index.Index,
        outlength: Integral,
        ascending: bool,
        stable: bool,
        kind: Any,
        order: Any,
    ):
        raise ak._util.error(NotImplementedError)

    def sort(
        self,
        axis: Integral = -1,
        ascending: bool = True,
        stable: bool = False,
        kind: Any = None,
        order: Any = None,
    ):
        negaxis = -axis
        branch, depth = self.branch_depth
        if branch:
            if negaxis <= 0:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves "
                        "of the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use axis={} on a nested list structure that splits into "
                        "different depths, the minimum of which is depth={} from the leaves".format(
                            axis, depth
                        )
                    )
                )
        else:
            if negaxis <= 0:
                negaxis = negaxis + depth
            if not (0 < negaxis and negaxis <= depth):
                raise ak._errors.wrap_error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
                )

        starts = ak.index.Index64.zeros(1, self._nplike)
        parents = ak.index.Index64.zeros(self.length, self._nplike)
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

    def _sort_next(
        self,
        negaxis: Integral,
        starts: ak.index.Index,
        parents: ak.index.Index,
        outlength: Integral,
        ascending: bool,
        stable: bool,
        kind: Any,
        order: Any,
    ):
        raise ak._util.error(NotImplementedError)

    def _combinations_axis0(
        self,
        n: Integral,
        replacement: bool,
        recordlookup: list[str] | None,
        parameters: dict | None,
    ):
        size = self.length
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

        tocarryraw = self._nplike.index_nplike.empty(n, dtype=np.intp)
        tocarry = []
        for i in range(n):
            ptr = ak.index.Index64.empty(combinationslen, self._nplike, dtype=np.int64)
            tocarry.append(ptr)
            if self._nplike.known_data:
                tocarryraw[i] = ptr.ptr

        toindex = ak.index.Index64.empty(n, self._nplike, dtype=np.int64)
        fromindex = ak.index.Index64.empty(n, self._nplike, dtype=np.int64)

        assert toindex.nplike is self._nplike and fromindex.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_RegularArray_combinations_64",
                np.int64,
                toindex.data.dtype.type,
                fromindex.data.dtype.type,
            ](
                tocarryraw,
                toindex.data,
                fromindex.data,
                n,
                replacement,
                self.length,
                1,
            )
        )
        contents = []
        length = None
        for ptr in tocarry:
            contents.append(ak.contents.IndexedArray(ptr, self, None, self._nplike))
            length = contents[-1].length
        assert length is not None
        return ak.contents.RecordArray(
            contents, recordlookup, length, parameters, self._nplike
        )

    def combinations(
        self,
        n: Integral,
        replacement: bool = False,
        axis: Integral = 1,
        fields: list[str] | None = None,
        parameters: dict | None = None,
    ):
        if n < 1:
            raise ak._errors.wrap_error(
                ValueError("in combinations, 'n' must be at least 1")
            )

        recordlookup = None
        if fields is not None:
            recordlookup = fields
            if len(recordlookup) != n:
                raise ak._errors.wrap_error(
                    ValueError("if provided, the length of 'fields' must be 'n'")
                )
        return self._combinations(n, replacement, recordlookup, parameters, axis, 0)

    def _combinations(
        self,
        n: Integral,
        replacement: bool,
        recordlookup: list[str] | None,
        parameters: dict[str, Any] | None,
        axis: Integral,
        depth: Integral,
    ):
        raise ak._util.error(NotImplementedError)

    def validity_error_parameters(self, path: str) -> str:
        if self.parameter("__array__") == "string":
            content = None
            if isinstance(
                self,
                (
                    ak.contents.ListArray,
                    ak.contents.ListOffsetArray,
                    ak.contents.RegularArray,
                ),
            ):
                content = self.content
            else:
                return 'at {} ("{}"): __array__ = "string" only allowed for ListArray, ListOffsetArray and RegularArray'.format(
                    path, type(self)
                )
            if content.parameter("__array__") != "char":
                return 'at {} ("{}"): __array__ = "string" must directly contain a node with __array__ = "char"'.format(
                    path, type(self)
                )
            if isinstance(content, ak.contents.NumpyArray):
                if content.dtype.type != np.uint8:
                    return 'at {} ("{}"): __array__ = "char" requires dtype == uint8'.format(
                        path, type(self)
                    )
                if len(content.shape) != 1:
                    return 'at {} ("{}"): __array__ = "char" must be one-dimensional'.format(
                        path, type(self)
                    )
            else:
                return 'at {} ("{}"): __array__ = "char" only allowed for NumpyArray'.format(
                    path, type(self)
                )
            return ""

        if self.parameter("__array__") == "bytestring":
            content = None
            if isinstance(
                self,
                (
                    ak.contents.ListArray,
                    ak.contents.ListOffsetArray,
                    ak.contents.RegularArray,
                ),
            ):
                content = self.content
            else:
                return 'at {} ("{}"): __array__ = "bytestring" only allowed for ListArray, ListOffsetArray and RegularArray'.format(
                    path, type(self)
                )
            if content.parameter("__array__") != "byte":
                return 'at {} ("{}"): __array__ = "bytestring" must directly contain a node with __array__ = "byte"'.format(
                    path, type(self)
                )
            if isinstance(content, ak.contents.NumpyArray):
                if content.dtype.type != np.uint8:
                    return 'at {} ("{}"): __array__ = "byte" requires dtype == uint8'.format(
                        path, type(self)
                    )
                if len(content.shape) != 1:
                    return 'at {} ("{}"): __array__ = "byte" must be one-dimensional'.format(
                        path, type(self)
                    )
            else:
                return 'at {} ("{}"): __array__ = "byte" only allowed for NumpyArray'.format(
                    path, type(self)
                )
            return ""

        if self.parameter("__array__") == "char":
            return 'at {} ("{}"): __array__ = "char" must be directly inside __array__ = "string"'.format(
                path, type(self)
            )

        if self.parameter("__array__") == "byte":
            return 'at {} ("{}"): __array__ = "byte" must be directly inside __array__ = "bytestring"'.format(
                path, type(self)
            )

        if self.parameter("__array__") == "categorical":
            content = None
            if isinstance(
                self,
                (
                    ak.contents.IndexedArray,
                    ak.contents.IndexedOptionArray,
                ),
            ):
                content = self.content
            else:
                return 'at {} ("{}"): __array__ = "categorical" only allowed for IndexedArray and IndexedOptionArray'.format(
                    path, type(self)
                )
            if not content.is_unique():
                return 'at {} ("{}"): __array__ = "categorical" requires contents to be unique'.format(
                    path, type(self)
                )

        return ""

    def validity_error(self, path: str = "layout") -> str:
        paramcheck = self.validity_error_parameters(path)
        if paramcheck != "":
            return paramcheck
        return self._validity_error(path)

    def _validity_error(self, path: str) -> str:
        raise ak._util.error(NotImplementedError)

    @property
    def nbytes(self) -> int:
        return self._nbytes_part()

    def purelist_parameter(self, key: str):
        return self.Form.purelist_parameter(self, key)

    def is_unique(self, axis: Integral | None = None) -> bool:
        negaxis = axis if axis is None else -axis
        starts = ak.index.Index64.zeros(1, self._nplike)
        parents = ak.index.Index64.zeros(self.length, self._nplike)
        return self._is_unique(negaxis, starts, parents, 1)

    def _is_unique(
        self,
        negaxis: Integral | None,
        starts: ak.index.Index,
        parents: ak.index.Index,
        outlength: Integral,
    ) -> bool:
        raise ak._util.error(NotImplementedError)

    def unique(self, axis=None):
        if axis == -1 or axis is None:
            negaxis = axis if axis is None else -axis
            if negaxis is not None:
                branch, depth = self.branch_depth
                if branch:
                    if negaxis <= 0:
                        raise ak._errors.wrap_error(
                            np.AxisError(
                                "cannot use non-negative axis on a nested list structure "
                                "of variable depth (negative axis counts from the leaves "
                                "of the tree; non-negative from the root)"
                            )
                        )
                    if negaxis > depth:
                        raise ak._errors.wrap_error(
                            np.AxisError(
                                "cannot use axis={} on a nested list structure that splits into "
                                "different depths, the minimum of which is depth={} from the leaves".format(
                                    axis, depth
                                )
                            )
                        )
                else:
                    if negaxis <= 0:
                        negaxis = negaxis + depth
                    if not (0 < negaxis and negaxis <= depth):
                        raise ak._errors.wrap_error(
                            np.AxisError(
                                "axis={} exceeds the depth of this array ({})".format(
                                    axis, depth
                                )
                            )
                        )

            starts = ak.index.Index64.zeros(1, self._nplike)
            parents = ak.index.Index64.zeros(self.length, self._nplike)

            return self._unique(negaxis, starts, parents, 1)

        raise ak._errors.wrap_error(
            np.AxisError(
                "unique expects axis 'None' or '-1', got axis={} that is not supported yet".format(
                    axis
                )
            )
        )

    def _unique(
        self,
        negaxis: Integral | None,
        starts: ak.index.Index,
        parents: ak.index.Index,
        outlength: Integral,
    ):
        raise ak._util.error(NotImplementedError)

    @property
    def is_identity_like(self) -> bool:
        return self.Form.is_identity_like.__get__(self)

    @property
    def purelist_isregular(self) -> bool:
        return self.Form.purelist_isregular.__get__(self)

    @property
    def purelist_depth(self) -> int:
        return self.Form.purelist_depth.__get__(self)

    @property
    def minmax_depth(self) -> tuple[int, int]:
        return self.Form.minmax_depth.__get__(self)

    @property
    def branch_depth(self) -> tuple[bool, int]:
        return self.Form.branch_depth.__get__(self)

    @property
    def fields(self) -> list[str]:
        return self.Form.fields.__get__(self)

    @property
    def is_tuple(self) -> bool:
        return self.Form.is_tuple.__get__(self)

    @property
    def dimension_optiontype(self) -> bool:
        return self.Form.dimension_optiontype.__get__(self)

    def pad_none_axis0(self, target: Integral, clip: bool) -> Content:
        if not clip and target < self.length:
            index = ak.index.Index64(
                self._nplike.index_nplike.arange(self.length, dtype=np.int64),
                nplike=self.nplike,
            )

        else:
            index = ak.index.Index64.empty(target, self._nplike)

            assert index.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_index_rpad_and_clip_axis0",
                    index.dtype.type,
                ](index.data, target, self.length)
            )

        next = ak.contents.IndexedOptionArray(
            index,
            self,
            self._parameters,
            self._nplike,
        )
        return next.simplify_optiontype()

    def pad_none(self, length: Integral, axis: Integral, clip: bool = False) -> Content:
        return self._pad_none(length, axis, 0, clip)

    def _pad_none(
        self, target: Integral, axis: Integral, depth: Integral, clip: bool
    ) -> Content:
        raise ak._util.error(NotImplementedError)

    def to_arrow(
        self,
        list_to32: bool = False,
        string_to32: bool = False,
        bytestring_to32: bool = False,
        emptyarray_to=None,
        categorical_as_dictionary: bool = False,
        extensionarray: bool = True,
        count_nulls: bool = True,
        record_is_scalar: bool = False,
    ):
        import awkward._connect.pyarrow

        pyarrow = awkward._connect.pyarrow.import_pyarrow("to_arrow")
        return self._to_arrow(
            pyarrow,
            None,
            None,
            self.length,
            {
                "list_to32": list_to32,
                "string_to32": string_to32,
                "bytestring_to32": bytestring_to32,
                "emptyarray_to": emptyarray_to,
                "categorical_as_dictionary": categorical_as_dictionary,
                "extensionarray": extensionarray,
                "count_nulls": count_nulls,
                "record_is_scalar": record_is_scalar,
            },
        )

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Any,
        validbytes: Any,
        length: Integral,
        options: dict[str, Any],
    ):
        raise ak._util.error(NotImplementedError)

    def to_numpy(self, allow_missing: bool):
        return self._to_numpy(allow_missing)

    def _to_numpy(self, allow_missing: bool):
        raise ak._util.error(NotImplementedError)

    def completely_flatten(
        self,
        nplike: NumpyLike | None = None,
        flatten_records: bool = True,
        function_name: str | None = None,
        drop_nones: bool = True,
    ):
        if nplike is None:
            nplike = self._nplike
        arrays = self._completely_flatten(
            nplike,
            {
                "flatten_records": flatten_records,
                "function_name": function_name,
                "drop_nones": drop_nones,
            },
        )
        return tuple(arrays)

    def _completely_flatten(
        self, nplike: NumpyLike | None, options: dict[str, Any]
    ) -> list:
        raise ak._util.error(NotImplementedError)

    def recursively_apply(
        self,
        action: ActionType,
        behavior: dict | None = None,
        depth_context: dict[str, Any] | None = None,
        lateral_context: dict[str, Any] | None = None,
        allow_records: bool = True,
        keep_parameters: bool = True,
        numpy_to_regular: bool = True,
        return_array: bool = True,
        function_name: str | None = None,
    ) -> Content | None:
        return self._recursively_apply(
            action,
            behavior,
            1,
            copy.copy(depth_context),
            lateral_context,
            {
                "allow_records": allow_records,
                "keep_parameters": keep_parameters,
                "numpy_to_regular": numpy_to_regular,
                "return_array": return_array,
                "function_name": function_name,
            },
        )

    def _recursively_apply(
        self,
        action: ActionType,
        behavior: dict | None,
        depth: Integral,
        depth_context: dict | None,
        lateral_context: dict | None,
        options: dict[str, Any],
    ) -> Content | None:
        raise ak._util.error(NotImplementedError)

    def to_json(
        self,
        nan_string: str | None = None,
        posinf_string: str | None = None,
        neginf_string: str | None = None,
        complex_record_fields: tuple[str, str] | None = None,
        convert_bytes: bool | None = None,
        behavior: dict | None = None,
    ) -> list:
        if complex_record_fields is None:
            complex_real_string = None
            complex_imag_string = None
        elif (
            isinstance(complex_record_fields, Sized)
            and isinstance(complex_record_fields, Iterable)
            and len(complex_record_fields) == 2
            and isinstance(complex_record_fields[0], str)
            and isinstance(complex_record_fields[1], str)
        ):
            complex_real_string, complex_imag_string = complex_record_fields
        else:
            complex_real_string, complex_imag_string = None, None

        return self.packed()._to_list(
            behavior,
            {
                "nan_string": nan_string,
                "posinf_string": posinf_string,
                "neginf_string": neginf_string,
                "complex_real_string": complex_real_string,
                "complex_imag_string": complex_imag_string,
                "convert_bytes": convert_bytes,
            },
        )

    def packed(self) -> Content:
        raise ak._util.error(NotImplementedError)

    def tolist(self, behavior: dict | None = None) -> list:
        return self.to_list(behavior)

    def to_list(self, behavior: dict | None = None) -> list:
        return self.packed()._to_list(behavior, None)

    def _to_list(
        self, behavior: dict | None, json_conversions: dict[str, Any] | None
    ) -> list:
        raise ak._util.error(NotImplementedError)

    def _to_list_custom(
        self, behavior: dict | None, json_conversions: dict[str, Any] | None
    ):
        if self.is_record:
            getitem = ak._util.recordclass(self, behavior).__getitem__
            overloaded = getitem is not ak.highlevel.Record.__getitem__ and not getattr(
                getitem, "ignore_in_to_list", False
            )
        else:
            getitem = ak._util.arrayclass(self, behavior).__getitem__
            overloaded = getitem is not ak.highlevel.Array.__getitem__ and not getattr(
                getitem, "ignore_in_to_list", False
            )

        if overloaded:
            array = ak._util.wrap(self, behavior=behavior)
            out = [None] * self.length
            for i in range(self.length):
                out[i] = array[i]
                if isinstance(out[i], (ak.highlevel.Array, ak.highlevel.Record)):
                    out[i] = out[i]._layout._to_list(behavior, json_conversions)
                elif hasattr(out[i], "tolist"):
                    out[i] = out[i].tolist()

            # These json_conversions are applied in NumpyArray (for numbers)
            # and ListArray/ListOffsetArray/RegularArray (for bytestrings),
            # but they're also applied here because __getitem__ might return
            # something convertible (the overloaded __getitem__ might be
            # trivial, as it is in Vector).
            if json_conversions is not None:
                convert_bytes = json_conversions["convert_bytes"]
                if convert_bytes is not None:
                    for i, x in enumerate(out):
                        if isinstance(x, bytes):
                            out[i] = convert_bytes(x)

                outimag = None
                complex_real_string = json_conversions["complex_real_string"]
                complex_imag_string = json_conversions["complex_imag_string"]
                if complex_real_string is not None:
                    if any(
                        not isinstance(x, Real) and isinstance(x, Complex) for x in out
                    ):
                        outimag = [None] * len(out)
                        for i, x in enumerate(out):
                            if isinstance(x, Complex):
                                out[i] = x.real
                                outimag[i] = x.imag
                            else:
                                out[i] = x
                                outimag[i] = None

                filters = []

                nan_string = json_conversions["nan_string"]
                if nan_string is not None:
                    isnan = math.isnan
                    filters.append(lambda x: nan_string if isnan(x) else x)

                posinf_string = json_conversions["posinf_string"]
                if posinf_string is not None:
                    inf = float("inf")
                    filters.append(lambda x: posinf_string if x == inf else x)

                neginf_string = json_conversions["neginf_string"]
                if neginf_string is not None:
                    minf = float("-inf")
                    filters.append(lambda x: neginf_string if x == minf else x)

                if len(filters) == 1:
                    f0 = filters[0]
                    for i, x in enumerate(out):
                        out[i] = f0(x)
                    if outimag is not None:
                        for i, x in enumerate(outimag):
                            outimag[i] = f0(x)
                elif len(filters) == 2:
                    f0 = filters[0]
                    f1 = filters[1]
                    for i, x in enumerate(out):
                        out[i] = f1(f0(x))
                    if outimag is not None:
                        for i, x in enumerate(outimag):
                            outimag[i] = f1(f0(x))
                elif len(filters) == 3:
                    f0 = filters[0]
                    f1 = filters[1]
                    f2 = filters[2]
                    for i, x in enumerate(out):
                        out[i] = f2(f1(f0(x)))
                    if outimag is not None:
                        for i, x in enumerate(outimag):
                            outimag[i] = f2(f1(f0(x)))

                if outimag is not None:
                    for i, (real, imag) in enumerate(zip(out, outimag)):
                        out[i] = {complex_real_string: real, complex_imag_string: imag}

            return out

    def flatten(self, axis: Integral = 1, depth: Integral = 0) -> Content:
        offsets, flattened = self._offsets_and_flattened(axis, depth)
        return flattened

    def _offsets_and_flattened(
        self, axis: Integral, depth: Integral
    ) -> tuple[ak.index.Index, Content]:
        raise ak._util.error(NotImplementedError)

    def to_backend(self, backend: str) -> Self:
        if self.nplike is ak._util.regularize_backend(backend):
            return self
        else:
            return self._to_nplike(ak._util.regularize_backend(backend))

    def _to_nplike(self, nplike: NumpyLike) -> Self:
        raise ak._util.error(NotImplementedError)

    def with_parameter(self, key: str, value: Any) -> Self:
        out = copy.copy(self)

        if self._parameters is None:
            parameters = {}
        else:
            parameters = copy.copy(self._parameters)
        parameters[key] = value
        out._parameters = parameters

        return out

    def __copy__(self):
        raise ak._errors.wrap_error(NotImplementedError)

    def __deepcopy__(self, memo):
        raise ak._errors.wrap_error(NotImplementedError)

    def layout_equal(
        self, other: Content, index_dtype: bool = True, numpyarray: bool = True
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and len(self) == len(other)
            and ak.forms.form._parameters_equal(
                self.parameters, other.parameters, only_array_record=False
            )
            and self._layout_equal(other, index_dtype, numpyarray)
        )

    def _layout_equal(
        self, other: Self, index_dtype: bool = True, numpyarray: bool = True
    ) -> bool:
        raise ak._util.error(NotImplementedError)

    def _repr(self, indent: str, pre: str, post: str) -> str:
        raise ak._util.error(NotImplementedError)
