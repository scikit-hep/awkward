# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import awkward as ak
from awkward._v2.contents.content import Content, unset
from awkward._v2.forms.emptyform import EmptyForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class EmptyArray(Content):
    is_NumpyType = True
    is_UnknownType = True

    def copy(
        self,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return EmptyArray(
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(self, identifier=None, parameters=None, nplike=None):
        if nplike is None:
            nplike = numpy
        self._init(identifier, parameters, nplike)

    Form = EmptyForm

    def _form_with_key(self, getkey):
        return self.Form(
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=getkey(self),
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return EmptyArray(self._typetracer_identifier(), self._parameters, tt)

    @property
    def length(self):
        return 0

    def _forget_length(self):
        return EmptyArray(self._identifier, self._parameters, self._nplike)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        extra = self._repr_extra(indent + "    ")
        if len(extra) == 0:
            return indent + pre + "<EmptyArray len='0'/>" + post
        else:
            out = [indent, pre, "<EmptyArray len='0'>"]
            out.extend(extra)
            out.append("\n" + indent + "</EmptyArray>")
            out.append(post)
            return "".join(out)

    def merge_parameters(self, parameters):
        return EmptyArray(
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toNumpyArray(self, dtype, nplike=None):
        if nplike is None:
            nplike = self._nplike
        if nplike is None:
            nplike = numpy
        return ak._v2.contents.numpyarray.NumpyArray(
            nplike.empty(0, dtype), self._identifier, self._parameters, nplike
        )

    def __array__(self, **kwargs):
        return numpy.empty((0,))

    def __iter__(self):
        return iter([])

    def _getitem_nothing(self):
        return self

    def _getitem_at(self, where):
        raise ak._v2._util.indexerror(self, where, "array is empty")

    def _getitem_range(self, where):
        return self

    def _getitem_field(self, where, only_fields=()):
        raise ak._v2._util.indexerror(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise ak._v2._util.indexerror(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        if not carry.nplike.known_shape or carry.length == 0:
            return self
        else:
            raise ak._v2._util.indexerror(self, carry.data, "array is empty")

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        raise ak._v2._util.indexerror(
            self,
            ak._v2.contents.ListArray(
                slicestarts, slicestops, slicecontent, None, None, self._nplike
            ),
            "too many jagged slice dimensions for array",
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            raise ak._v2._util.indexerror(self, head, "array is empty")

        elif isinstance(head, slice):
            raise ak._v2._util.indexerror(self, head, "array is empty")

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            if not head.nplike.known_shape or head.length == 0:
                return self
            else:
                raise ak._v2._util.indexerror(self, head.data, "array is empty")

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise ak._v2._util.indexerror(self, head, "array is empty")

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise ak._v2._util.indexerror(self, head, "array is empty")

        else:
            raise ak._v2._util.error(AssertionError(repr(head)))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)

        if posaxis == depth:
            out = self.length
            if ak._v2._util.isint(out):
                return np.int64(out)
            else:
                return out
        else:
            out = ak._v2.index.Index64.empty(0, self._nplike)
            return ak._v2.contents.numpyarray.NumpyArray(out, None, None, self._nplike)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise ak._v2._util.error(
                np.AxisError(self, "axis=0 not allowed for flatten")
            )
        else:
            offsets = ak._v2.index.Index64.zeros(1, self._nplike)
            return (offsets, EmptyArray(None, self._parameters, self._nplike))

    def _mergeable(self, other, mergebool):
        return True

    def mergemany(self, others):
        if len(others) == 0:
            return self

        elif len(others) == 1:
            return others[0]

        else:
            tail_others = others[1:]
            return others[0].mergemany(tail_others)

    def fill_none(self, value):
        return EmptyArray(None, self._parameters, self._nplike)

    def _local_index(self, axis, depth):
        return ak._v2.contents.numpyarray.NumpyArray(
            self._nplike.empty(0, np.int64), None, None, self._nplike
        )

    def numbers_to_type(self, name):
        return ak._v2.contents.emptyarray.EmptyArray(
            self._identifier, self._parameters, self._nplike
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        return True

    def _unique(self, negaxis, starts, parents, outlength):
        return self

    def _argsort_next(
        self,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        ascending,
        stable,
        kind,
        order,
    ):
        as_numpy = self.toNumpyArray(np.float64)
        return as_numpy._argsort_next(
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        return self

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return ak._v2.contents.emptyarray.EmptyArray(
            self._identifier, self._parameters, self._nplike
        )

    def _reduce_next(
        self,
        reducer,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        mask,
        keepdims,
        behavior,
    ):
        as_numpy = self.toNumpyArray(reducer.preferred_dtype)
        return as_numpy._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            mask,
            keepdims,
            behavior,
        )

    def _validity_error(self, path):
        return ""

    def _nbytes_part(self):
        if self.identifier is not None:
            return self.identifier._nbytes_part()
        return 0

    def _pad_none(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis != depth:
            raise ak._v2._util.error(
                np.AxisError(f"axis={axis} exceeds the depth of this array({depth})")
            )
        else:
            return self.pad_none_axis0(target, True)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if options["emptyarray_to"] is None:
            return pyarrow.Array.from_buffers(
                ak._v2._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.null(),
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                length,
                [
                    ak._v2._connect.pyarrow.to_validbits(validbytes),
                ],
                null_count=length,
            )

        else:
            dtype = np.dtype(options["emptyarray_to"])
            next = ak._v2.contents.numpyarray.NumpyArray(
                numpy.empty(length, dtype),
                self._identifier,
                self._parameters,
                nplike=self.nplike,
            )
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

    def _to_numpy(self, allow_missing):
        return self._nplike.empty(0, dtype=np.float64)

    def _completely_flatten(self, nplike, options):
        return []

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return EmptyArray(self._identifier, None, self._nplike)

        else:

            def continuation():
                pass

        result = action(
            self,
            depth=depth,
            depth_context=depth_context,
            lateral_context=lateral_context,
            continuation=continuation,
            behavior=behavior,
            nplike=self._nplike,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._v2._util.error(AssertionError(result))

    def packed(self):
        return self

    def _to_list(self, behavior, json_conversions):
        return []

    def _to_nplike(self, nplike):
        return EmptyArray(self._identifier, self._parameters, nplike=nplike)

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return True
