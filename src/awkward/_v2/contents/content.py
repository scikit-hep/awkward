# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers
import math
import copy
from collections.abc import Sized, Iterable

import awkward as ak
import awkward._v2._reducers
from awkward._v2.tmp_for_testing import v1_to_v2

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()

# FIXME: use common Sentinel class for this
unset = object()


class Content:
    is_NumpyType = False
    is_UnknownType = False
    is_ListType = False
    is_RegularType = False
    is_OptionType = False
    is_IndexedType = False
    is_RecordType = False
    is_UnionType = False

    def _init(self, identifier, parameters, nplike):
        if identifier is not None and not isinstance(
            identifier, ak._v2.identifier.Identifier
        ):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'identifier' must be an Identifier or None, not {}".format(
                        type(self).__name__, repr(identifier)
                    )
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'parameters' must be a dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )

        if nplike is not None and not isinstance(nplike, ak.nplike.NumpyLike):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'nplike' must be an ak.nplike.NumpyLike or None, not {}".format(
                        type(self).__name__, repr(nplike)
                    )
                )
            )

        self._identifier = identifier
        self._parameters = parameters
        self._nplike = nplike

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

    @property
    def nplike(self):
        return self._nplike

    @property
    def form(self):
        return self.form_with_key(None)

    def form_with_key(self, form_key="node{id}", id_start=0):
        hold_id = [id_start]

        if form_key is None:

            def getkey(layout):
                return None

        elif ak._v2._util.isstr(form_key):

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
            raise ak._v2._util.error(
                TypeError(
                    "form_key must be None, a string, or a callable, not {}".format(
                        type(form_key)
                    )
                )
            )

        return self._form_with_key(getkey)

    def forget_length(self):
        if not isinstance(self._nplike, ak._v2._typetracer.TypeTracer):
            return self.typetracer._forget_length()
        else:
            return self._forget_length()

    def to_buffers(
        self,
        container=None,
        buffer_key="{form_key}-{attribute}",
        form_key="node{id}",
        id_start=0,
        nplike=None,
    ):
        if container is None:
            container = {}
        if nplike is None:
            nplike = self._nplike
        if not nplike.known_data:
            raise ak._v2._util.error(
                TypeError("cannot call 'to_buffers' on an array without concrete data")
            )

        if ak._v2._util.isstr(buffer_key):

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
            raise ak._v2._util.error(
                TypeError(
                    "buffer_key must be a string or a callable, not {}".format(
                        type(buffer_key)
                    )
                )
            )

        if form_key is None:
            raise ak._v2._util.error(
                TypeError(
                    "a 'form_key' must be supplied, to match Form elements to buffers in the 'container'"
                )
            )

        form = self.form_with_key(form_key=form_key, id_start=id_start)

        self._to_buffers(form, getkey, container, nplike)

        return form, len(self), container

    def __len__(self):
        return self.length

    def _repr_extra(self, indent):
        out = []
        if self._parameters is not None:
            for k, v in self._parameters.items():
                out.append(
                    "\n{}<parameter name={}>{}</parameter>".format(
                        indent, repr(k), repr(v)
                    )
                )
        if self._identifier is not None:
            out.append(self._identifier._repr("\n" + indent, "", ""))
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
                raise ak._v2._util.error(ValueError(message + filename))

            else:
                if error.id != ak._util.kSliceNone and self._identifier is not None:
                    # FIXME https://github.com/scikit-hep/awkward-1.0/blob/45d59ef4ae45eebb02995b8e1acaac0d46fb9573/src/libawkward/util.cpp#L443-L450
                    pass

                if error.attempt != ak._util.kSliceNone:
                    message += f" while attempting to get index {error.attempt}"

                message += filename

                if slicer is None:
                    raise ak._v2._util.error(ValueError(message))
                else:
                    raise ak._v2._util.indexerror(self, slicer, message)

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
                raise ak._v2._util.error(ValueError(message + filename))

            else:
                if error.attempt != ak._util.kSliceNone:
                    message += f" while attempting to get index {error.attempt}"

                message += filename

                raise ak._v2._util.error(ValueError(message))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise ak._v2._util.error(
            TypeError(
                "do not apply NumPy functions to low-level layouts (Content subclasses); put them in ak._v2.highlevel.Array"
            )
        )

    def __array_function__(self, func, types, args, kwargs):
        raise ak._v2._util.error(
            TypeError(
                "do not apply NumPy functions to low-level layouts (Content subclasses); put them in ak._v2.highlevel.Array"
            )
        )

    def __array__(self, **kwargs):
        raise ak._v2._util.error(
            TypeError(
                "do not try to convert low-level layouts (Content subclasses) into NumPy arrays; put them in ak._v2.highlevel.Array"
            )
        )

    def __iter__(self):
        if not self._nplike.known_data:
            raise ak._v2._util.error(
                TypeError("cannot iterate on an array without concrete data")
            )

        for i in range(len(self)):
            yield self._getitem_at(i)

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
            self._nplike,
        )

    def _getitem_next_ellipsis(self, tail, advanced):
        mindepth, maxdepth = self.minmax_depth

        dimlength = sum(
            1 if isinstance(x, (int, slice, ak._v2.index.Index64)) else 0 for x in tail
        )

        if len(tail) == 0 or mindepth - 1 == maxdepth - 1 == dimlength:
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            return self._getitem_next(nexthead, nexttail, advanced)

        elif dimlength in {mindepth - 1, maxdepth - 1}:
            raise ak._v2._util.indexerror(
                self,
                Ellipsis,
                "ellipsis (`...`) can't be used on data with different numbers of dimensions",
            )

        else:
            return self._getitem_next(slice(None), (Ellipsis,) + tail, advanced)

    def _getitem_next_regular_missing(self, head, tail, advanced, raw, length):
        # if this is in a tuple-slice and really should be 0, it will be trimmed later
        length = 1 if length == 0 else length
        index = ak._v2.index.Index64(head.index, nplike=self.nplike)
        indexlength = index.length
        index = index._to_nplike(self.nplike)
        outindex = ak._v2.index.Index64.empty(index.length * length, self._nplike)

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

        out = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outindex, raw.content, None, self._parameters, self._nplike
        )

        return ak._v2.contents.regulararray.RegularArray(
            out.simplify_optiontype(),
            indexlength,
            1,
            None,
            self._parameters,
            self._nplike,
        )

    def _getitem_next_missing_jagged(self, head, tail, advanced, that):
        head = head._to_nplike(self._nplike)
        jagged = head.content.toListOffsetArray64()

        index = ak._v2.index.Index64(head._index, nplike=self.nplike)
        content = that._getitem_at(0)
        if self._nplike.known_shape and content.length < index.length:
            raise ak._v2._util.indexerror(
                self,
                head,
                "cannot fit masked jagged slice with length {} into {} of size {}".format(
                    index.length, type(that).__name__, content.length
                ),
            )

        outputmask = ak._v2.index.Index64.empty(index.length, self._nplike)
        starts = ak._v2.index.Index64.empty(index.length, self._nplike)
        stops = ak._v2.index.Index64.empty(index.length, self._nplike)

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
        out = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outputmask, tmp, None, self._parameters, self._nplike
        )
        return ak._v2.contents.regulararray.RegularArray(
            out.simplify_optiontype(),
            index.length,
            1,
            None,
            self._parameters,
            self._nplike,
        )

    def _getitem_next_missing(self, head, tail, advanced):
        assert isinstance(head, ak._v2.contents.IndexedOptionArray)

        if advanced is not None:
            raise ak._v2._util.indexerror(
                self,
                head,
                "cannot mix missing values in slice with NumPy-style advanced indexing",
            )

        if isinstance(head.content, ak._v2.contents.listoffsetarray.ListOffsetArray):
            if self.nplike.known_shape and self.length != 1:
                raise ak._v2._util.error(
                    NotImplementedError("reached a not-well-considered code path")
                )
            return self._getitem_next_missing_jagged(head, tail, advanced, self)

        if isinstance(head.content, ak._v2.contents.numpyarray.NumpyArray):
            headcontent = ak._v2.index.Index64(head.content.data)
            nextcontent = self._getitem_next(headcontent, tail, advanced)
        else:
            nextcontent = self._getitem_next(head.content, tail, advanced)

        if isinstance(nextcontent, ak._v2.contents.regulararray.RegularArray):
            return self._getitem_next_regular_missing(
                head, tail, advanced, nextcontent, nextcontent.length
            )

        elif isinstance(nextcontent, ak._v2.contents.recordarray.RecordArray):
            if len(nextcontent._fields) == 0:
                return nextcontent

            contents = []

            for content in nextcontent.contents:
                if isinstance(content, ak._v2.contents.regulararray.RegularArray):
                    contents.append(
                        self._getitem_next_regular_missing(
                            head, tail, advanced, content, content.length
                        )
                    )
                else:
                    raise ak._v2._util.error(
                        NotImplementedError(
                            "FIXME: unhandled case of SliceMissing with RecordArray containing {}".format(
                                content
                            )
                        )
                    )

            return ak._v2.contents.recordarray.RecordArray(
                contents,
                nextcontent._fields,
                None,
                None,
                self._parameters,
                self._nplike,
            )

        else:
            raise ak._v2._util.error(
                NotImplementedError(
                    f"FIXME: unhandled case of SliceMissing with {nextcontent}"
                )
            )

    def __getitem__(self, where):
        return self._getitem(where)

    def _getitem(self, where):
        if ak._util.isint(where):
            return self._getitem_at(where)

        elif isinstance(where, slice) and where.step is None:
            return self._getitem_range(where)

        elif isinstance(where, slice):
            return self._getitem((where,))

        elif ak._util.isstr(where):
            return self._getitem_field(where)

        elif where is np.newaxis:
            return self._getitem((where,))

        elif where is Ellipsis:
            return self._getitem((where,))

        elif isinstance(where, tuple):
            if len(where) == 0:
                return self

            # Normalise valid indices onto well-defined basis
            items = ak._v2._slicing.normalise_items(where, self._nplike)
            # Prepare items for advanced indexing (e.g. via broadcasting)
            nextwhere = ak._v2._slicing.prepare_advanced_indexing(items)

            next = ak._v2.contents.RegularArray(
                self,
                self.length if self._nplike.known_shape else 1,
                1,
                None,
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

        elif isinstance(where, ak.layout.Content):
            return self._getitem(v1_to_v2(where))

        elif isinstance(where, ak._v2.highlevel.Array):
            return self._getitem(where.layout)

        elif (
            isinstance(where, Content)
            and where._parameters is not None
            and (where._parameters.get("__array__") in ("string", "bytestring"))
        ):
            return self._getitem_fields(ak._v2.operations.to_list(where))

        elif isinstance(where, ak._v2.contents.emptyarray.EmptyArray):
            return where.toNumpyArray(np.int64)

        elif isinstance(where, ak._v2.contents.numpyarray.NumpyArray):
            if issubclass(where.dtype.type, np.int64):
                carry = ak._v2.index.Index64(where.data.reshape(-1))
                allow_lazy = True
            elif issubclass(where.dtype.type, np.integer):
                carry = ak._v2.index.Index64(
                    where.data.astype(np.int64).reshape(-1), nplike=self.nplike
                )
                allow_lazy = "copied"  # True, but also can be modified in-place
            elif issubclass(where.dtype.type, (np.bool_, bool)):
                if len(where.data.shape) == 1:
                    where = self._nplike.nonzero(where.data)[0]
                    carry = ak._v2.index.Index64(where, nplike=self.nplike)
                    allow_lazy = "copied"  # True, but also can be modified in-place
                else:
                    wheres = self._nplike.nonzero(where.data)
                    return self._getitem(wheres)
            else:
                raise ak._v2._util.error(
                    TypeError(
                        "array slice must be an array of integers or booleans, not\n\n    {}".format(
                            repr(where.data).replace("\n", "\n    ")
                        )
                    )
                )

            out = ak._v2._slicing.getitem_next_array_wrap(
                self._carry(carry, allow_lazy), where.shape
            )
            if out.length == 0:
                return out._getitem_nothing()
            else:
                return out._getitem_at(0)

        elif isinstance(where, Content):
            return self._getitem((where,))

        elif ak._util.is_sized_iterable(where) and len(where) == 0:
            return self._carry(
                ak._v2.index.Index64.empty(0, self._nplike), allow_lazy=True
            )

        elif ak._util.is_sized_iterable(where) and all(
            ak._v2._util.isstr(x) for x in where
        ):
            return self._getitem_fields(where)

        elif ak._util.is_sized_iterable(where):
            layout = ak._v2.operations.to_layout(where)
            as_array = layout.maybe_to_array(layout.nplike)
            if as_array is None:
                return self._getitem(layout)
            else:
                return self._getitem(
                    ak._v2.contents.NumpyArray(as_array, None, None, layout.nplike)
                )

        else:
            raise ak._v2._util.error(
                TypeError(
                    "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                    "integer/boolean arrays (possibly with variable-length nested "
                    "lists or missing values), field name (str) or names (non-tuple "
                    "iterable of str) are valid indices for slicing, not\n\n    "
                    + repr(where).replace("\n", "\n    ")
                )
            )

    def _carry_asrange(self, carry):
        assert isinstance(carry, ak._v2.index.Index)

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
                raise ak._v2._util.error(IndexError)
        else:
            return None

    def _typetracer_identifier(self):
        if self._identifier is None:
            return None
        else:
            raise ak._v2._util.error(NotImplementedError)

    def _range_identifier(self, start, stop):
        if self._identifier is None:
            return None
        else:
            raise ak._v2._util.error(NotImplementedError)

    def _field_identifier(self, field):
        if self._identifier is None:
            return None
        else:
            raise ak._v2._util.error(NotImplementedError)

    def _fields_identifier(self, fields):
        if self._identifier is None:
            return None
        else:
            raise ak._v2._util.error(NotImplementedError)

    def _carry_identifier(self, carry):
        if self._identifier is None:
            return None
        else:
            raise ak._v2._util.error(NotImplementedError)

    def axis_wrap_if_negative(self, axis):
        if axis is None or axis >= 0:
            return axis

        mindepth, maxdepth = self.minmax_depth
        depth = self.purelist_depth
        if mindepth == depth and maxdepth == depth:
            posaxis = depth + axis
            if posaxis < 0:
                raise ak._v2._util.error(
                    np.AxisError(
                        f"axis={axis} exceeds the depth ({depth}) of this array"
                    )
                )
            return posaxis

        elif mindepth + axis == 0:
            raise ak._v2._util.error(
                np.AxisError(
                    "axis={} exceeds the depth ({}) of at least one record field (or union possibility) of this array".format(
                        axis, depth
                    )
                )
            )

        return axis

    def _local_index_axis0(self):
        localindex = ak._v2.index.Index64.empty(self.length, self._nplike)
        self._handle_error(
            self._nplike["awkward_localindex", np.int64](
                localindex.data,
                localindex.length,
            )
        )
        return ak._v2.contents.NumpyArray(localindex, None, None, self._nplike)

    def merge(self, other):
        others = [other]
        return self.mergemany(others)

    def mergeable(self, other, mergebool=True):
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_UnionType:
            return True
        # Otherwise, do the parameters match? If not, we can't merge.
        elif not (
            ak._v2.forms.form._parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            )
        ):
            return False
        # Finally, fall back upon the per-content implementation
        else:
            return self._mergeable(other, mergebool)

    def _mergeable(self, other: "Content", mergebool: bool) -> bool:
        raise ak._v2._util.error(NotImplementedError)

    def mergemany(self, others):
        raise ak._v2._util.error(NotImplementedError)

    def merge_as_union(self, other):
        mylength = self.length
        theirlength = other.length
        tags = ak._v2.index.Index8.empty((mylength + theirlength), self._nplike)
        index = ak._v2.index.Index64.empty((mylength + theirlength), self._nplike)
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

        return ak._v2.contents.unionarray.UnionArray(
            tags, index, contents, None, None, self._nplike
        )

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ak._v2._util.error(
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
                    ak._v2.contents.indexedarray.IndexedArray,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                    ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                    ak._v2.contents.bitmaskedarray.BitMaskedArray,
                    ak._v2.contents.unmaskedarray.UnmaskedArray,
                    ak._v2.contents.unionarray.UnionArray,
                ),
            ):
                break
            else:
                head.append(other)
            i = i + 1

        while i < len(others):
            tail.append(others[i])
            i = i + 1

        if any(
            isinstance(x.nplike, ak._v2._typetracer.TypeTracer) for x in head + tail
        ):
            head = [
                x
                if isinstance(x.nplike, ak._v2._typetracer.TypeTracer)
                else x.typetracer
                for x in head
            ]
            tail = [
                x
                if isinstance(x.nplike, ak._v2._typetracer.TypeTracer)
                else x.typetracer
                for x in tail
            ]

        return (head, tail)

    def dummy(self):
        raise ak._v2._util.error(
            NotImplementedError(
                "FIXME: need to implement 'dummy', which makes an array of length 1 and an arbitrary value (0?) for this array type"
            )
        )

    def local_index(self, axis):
        return self._local_index(axis, 0)

    def _reduce(self, reducer, axis=-1, mask=True, keepdims=False, behavior=None):
        if axis is None:
            raise ak._v2._util.error(NotImplementedError)

        negaxis = -axis
        branch, depth = self.branch_depth

        if branch:
            if negaxis <= 0:
                raise ak._v2._util.error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves of "
                        "the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._v2._util.error(
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
                raise ak._v2._util.error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
                )

        starts = ak._v2.index.Index64.zeros(1, self._nplike)
        parents = ak._v2.index.Index64.zeros(self.length, self._nplike)
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

    def argmin(self, axis=-1, mask=True, keepdims=False, behavior=None):
        return self._reduce(
            awkward._v2._reducers.ArgMin, axis, mask, keepdims, behavior
        )

    def argmax(self, axis=-1, mask=True, keepdims=False, behavior=None):
        return self._reduce(
            awkward._v2._reducers.ArgMax, axis, mask, keepdims, behavior
        )

    def count(self, axis=-1, mask=False, keepdims=False, behavior=None):
        return self._reduce(awkward._v2._reducers.Count, axis, mask, keepdims, behavior)

    def count_nonzero(self, axis=-1, mask=False, keepdims=False, behavior=None):
        return self._reduce(
            awkward._v2._reducers.CountNonzero, axis, mask, keepdims, behavior
        )

    def sum(self, axis=-1, mask=False, keepdims=False, behavior=None):
        return self._reduce(awkward._v2._reducers.Sum, axis, mask, keepdims, behavior)

    def prod(self, axis=-1, mask=False, keepdims=False, behavior=None):
        return self._reduce(awkward._v2._reducers.Prod, axis, mask, keepdims, behavior)

    def any(self, axis=-1, mask=False, keepdims=False, behavior=None):
        return self._reduce(awkward._v2._reducers.Any, axis, mask, keepdims, behavior)

    def all(self, axis=-1, mask=False, keepdims=False, behavior=None):
        return self._reduce(awkward._v2._reducers.All, axis, mask, keepdims, behavior)

    def min(self, axis=-1, mask=True, keepdims=False, initial=None, behavior=None):
        return self._reduce(
            awkward._v2._reducers.Min(initial), axis, mask, keepdims, behavior
        )

    def max(self, axis=-1, mask=True, keepdims=False, initial=None, behavior=None):
        return self._reduce(
            awkward._v2._reducers.Max(initial), axis, mask, keepdims, behavior
        )

    def argsort(self, axis=-1, ascending=True, stable=False, kind=None, order=None):
        negaxis = -axis
        branch, depth = self.branch_depth
        if branch:
            if negaxis <= 0:
                raise ak._v2._util.error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves "
                        "of the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._v2._util.error(
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
                raise ak._v2._util.error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
                )

        starts = ak._v2.index.Index64.zeros(1, self._nplike)
        parents = ak._v2.index.Index64.zeros(self.length, self._nplike)
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
                raise ak._v2._util.error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves "
                        "of the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._v2._util.error(
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
                raise ak._v2._util.error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
                )

        starts = ak._v2.index.Index64.zeros(1, self._nplike)
        parents = ak._v2.index.Index64.zeros(self.length, self._nplike)
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
            ptr = ak._v2.index.Index64.empty(
                combinationslen, self._nplike, dtype=np.int64
            )
            tocarry.append(ptr)
            if self._nplike.known_data:
                tocarryraw[i] = ptr.ptr

        toindex = ak._v2.index.Index64.empty(n, self._nplike, dtype=np.int64)
        fromindex = ak._v2.index.Index64.empty(n, self._nplike, dtype=np.int64)

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
            contents.append(
                ak._v2.contents.IndexedArray(ptr, self, None, None, self._nplike)
            )
            length = contents[-1].length
        assert length is not None
        return ak._v2.contents.recordarray.RecordArray(
            contents, recordlookup, length, None, parameters, self._nplike
        )

    def combinations(self, n, replacement=False, axis=1, fields=None, parameters=None):
        if n < 1:
            raise ak._v2._util.error(
                ValueError("in combinations, 'n' must be at least 1")
            )

        recordlookup = None
        if fields is not None:
            recordlookup = fields
            if len(recordlookup) != n:
                raise ak._v2._util.error(
                    ValueError("if provided, the length of 'fields' must be 'n'")
                )
        return self._combinations(n, replacement, recordlookup, parameters, axis, 0)

    def validity_error_parameters(self, path):
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
                return 'at {} ("{}"): __array__ = "string" only allowed for ListArray, ListOffsetArray and RegularArray'.format(
                    path, type(self)
                )
            if content.parameter("__array__") != "char":
                return 'at {} ("{}"): __array__ = "string" must directly contain a node with __array__ = "char"'.format(
                    path, type(self)
                )
            if isinstance(content, ak._v2.contents.numpyarray.NumpyArray):
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
                    ak._v2.contents.listarray.ListArray,
                    ak._v2.contents.listoffsetarray.ListOffsetArray,
                    ak._v2.contents.regulararray.RegularArray,
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
            if isinstance(content, ak._v2.contents.numpyarray.NumpyArray):
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
                    ak._v2.contents.indexedarray.IndexedArray,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray,
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

    def validity_error(self, path="layout"):
        paramcheck = self.validity_error_parameters(path)
        if paramcheck != "":
            return paramcheck
        return self._validity_error(path)

    @property
    def nbytes(self):
        return self._nbytes_part()

    def purelist_parameter(self, key):
        return self.Form.purelist_parameter(self, key)

    def is_unique(self, axis=None):
        negaxis = axis if axis is None else -axis
        starts = ak._v2.index.Index64.zeros(1, self._nplike)
        parents = ak._v2.index.Index64.zeros(self.length, self._nplike)
        return self._is_unique(negaxis, starts, parents, 1)

    def unique(self, axis=None):
        if axis == -1 or axis is None:
            negaxis = axis if axis is None else -axis
            if negaxis is not None:
                branch, depth = self.branch_depth
                if branch:
                    if negaxis <= 0:
                        raise ak._v2._util.error(
                            np.AxisError(
                                "cannot use non-negative axis on a nested list structure "
                                "of variable depth (negative axis counts from the leaves "
                                "of the tree; non-negative from the root)"
                            )
                        )
                    if negaxis > depth:
                        raise ak._v2._util.error(
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
                        raise ak._v2._util.error(
                            np.AxisError(
                                "axis={} exceeds the depth of this array ({})".format(
                                    axis, depth
                                )
                            )
                        )

            starts = ak._v2.index.Index64.zeros(1, self._nplike)
            parents = ak._v2.index.Index64.zeros(self.length, self._nplike)

            return self._unique(negaxis, starts, parents, 1)

        raise ak._v2._util.error(
            np.AxisError(
                "unique expects axis 'None' or '-1', got axis={} that is not supported yet".format(
                    axis
                )
            )
        )

    @property
    def is_identity_like(self):
        return self.Form.is_identity_like.__get__(self)

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
    def fields(self):
        return self.Form.fields.__get__(self)

    @property
    def is_tuple(self):
        return self.Form.is_tuple.__get__(self)

    @property
    def dimension_optiontype(self):
        return self.Form.dimension_optiontype.__get__(self)

    def pad_none_axis0(self, target, clip):
        if not clip and target < self.length:
            index = ak._v2.index.Index64(
                self._nplike.index_nplike.arange(self.length, dtype=np.int64),
                nplike=self.nplike,
            )

        else:
            index = ak._v2.index.Index64.empty(target, self._nplike)

            assert index.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_index_rpad_and_clip_axis0",
                    index.dtype.type,
                ](index.data, target, self.length)
            )

        next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            index,
            self,
            None,
            self._parameters,
            self._nplike,
        )
        return next.simplify_optiontype()

    def pad_none(self, length, axis, clip=False):
        return self._pad_none(length, axis, 0, clip)

    def to_arrow(
        self,
        list_to32=False,
        string_to32=False,
        bytestring_to32=False,
        emptyarray_to=None,
        categorical_as_dictionary=False,
        extensionarray=True,
        count_nulls=True,
        record_is_scalar=False,
    ):
        import awkward._v2._connect.pyarrow

        pyarrow = awkward._v2._connect.pyarrow.import_pyarrow("to_arrow")
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

    def to_numpy(self, allow_missing):
        return self._to_numpy(allow_missing)

    def completely_flatten(self, nplike=None, flatten_records=True, function_name=None):
        if nplike is None:
            nplike = self._nplike
        arrays = self._completely_flatten(
            nplike,
            {
                "flatten_records": flatten_records,
                "function_name": function_name,
            },
        )
        return tuple(arrays)

    def recursively_apply(
        self,
        action,
        behavior=None,
        depth_context=None,
        lateral_context=None,
        allow_records=True,
        keep_parameters=True,
        numpy_to_regular=True,
        return_array=True,
        function_name=None,
    ):
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

    def to_json(
        self,
        nan_string=None,
        posinf_string=None,
        neginf_string=None,
        complex_record_fields=None,
        convert_bytes=None,
        behavior=None,
    ):
        if complex_record_fields is None:
            complex_real_string = None
            complex_imag_string = None
        elif (
            isinstance(complex_record_fields, Sized)
            and isinstance(complex_record_fields, Iterable)
            and len(complex_record_fields) == 2
            and ak._v2._util.isstr(complex_record_fields[0])
            and ak._v2._util.isstr(complex_record_fields[1])
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

    def tolist(self, behavior=None):
        return self.to_list(behavior)

    def to_list(self, behavior=None):
        return self.packed()._to_list(behavior, None)

    def _to_list_custom(self, behavior, json_conversions):
        if self.is_RecordType:
            getitem = ak._v2._util.recordclass(self, behavior).__getitem__
            overloaded = (
                getitem is not ak._v2.highlevel.Record.__getitem__
                and not getattr(getitem, "ignore_in_to_list", False)
            )
        else:
            getitem = ak._v2._util.arrayclass(self, behavior).__getitem__
            overloaded = (
                getitem is not ak._v2.highlevel.Array.__getitem__
                and not getattr(getitem, "ignore_in_to_list", False)
            )

        if overloaded:
            array = ak._v2._util.wrap(self, behavior=behavior)
            out = [None] * self.length
            for i in range(self.length):
                out[i] = array[i]
                if isinstance(
                    out[i], (ak._v2.highlevel.Array, ak._v2.highlevel.Record)
                ):
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
                    Real = numbers.Real
                    Complex = numbers.Complex
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

    def flatten(self, axis=1, depth=0):
        offsets, flattened = self._offsets_and_flattened(axis, depth)
        return flattened

    def to_backend(self, backend):
        if self.nplike is ak._v2._util.regularize_backend(backend):
            return self
        else:
            return self._to_nplike(ak._v2._util.regularize_backend(backend))

    def with_parameter(self, key, value):
        out = copy.copy(self)

        if self._parameters is None:
            parameters = {}
        else:
            parameters = copy.copy(self._parameters)
        parameters[key] = value
        out._parameters = parameters

        return out

    def __copy__(self):
        raise ak._v2._util.error(NotImplementedError)

    def __deepcopy__(self, memo):
        raise ak._v2._util.error(NotImplementedError)

    def _jax_flatten(self):
        from awkward._v2._connect.jax import _find_numpyarray_nodes, AuxData

        layout = ak._v2.operations.to_layout(self, allow_record=True, allow_other=False)

        numpyarray_nodes = _find_numpyarray_nodes(layout)
        return (numpyarray_nodes, AuxData(layout))

    @classmethod
    def jax_flatten(cls, array):
        assert type(array) is cls
        return array._jax_flatten()

    @classmethod
    def jax_unflatten(cls, aux_data, children):
        from awkward._v2._connect.jax import _replace_numpyarray_nodes

        return _replace_numpyarray_nodes(aux_data.layout, list(children))

    def layout_equal(self, other, index_dtype=True, numpyarray=True):
        return (
            self.__class__ is other.__class__
            and len(self) == len(other)
            and ak._v2.identifier._identifiers_equal(self.identifier, other.identifier)
            and ak._v2.forms.form._parameters_equal(
                self.parameters, other.parameters, only_array_record=False
            )
            and self._layout_equal(other, index_dtype, numpyarray)
        )
