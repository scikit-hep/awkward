# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.emptyform import EmptyForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class EmptyArray(Content):
    is_NumpyType = True
    is_UnknownType = True

    def __init__(self, identifier=None, parameters=None):
        self._init(identifier, parameters)

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
        return EmptyArray(self._typetracer_identifier(), self._parameters)

    @property
    def nplike(self):
        return ak.nplike.Numpy.instance()

    def __len__(self):
        return 0

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
        )

    def toNumpyArray(self, dtype, nplike=numpy):
        return ak._v2.contents.numpyarray.NumpyArray(
            nplike.empty(0, dtype), self._identifier, self._parameters, nplike=nplike
        )

    def _getitem_nothing(self):
        return self

    def _getitem_at(self, where):
        raise NestedIndexError(self, where, "array is empty")

    def _getitem_range(self, where):
        return self

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        if len(carry) == 0:
            return self
        else:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, "array is empty")
            else:
                raise exception("array is empty")

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            raise NestedIndexError(self, head, "array is empty")

        elif isinstance(head, slice):
            raise NestedIndexError(self, head, "array is empty")

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            raise NestedIndexError(self, head, "array is empty")

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NestedIndexError(self, head, "array is empty")

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NestedIndexError(self, head, "array is empty")

        else:
            raise AssertionError(repr(head))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)

        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self.nplike)
            out[0] = len(self)
            return ak._v2.contents.numpyarray.NumpyArray(out)[0]
        else:
            out = ak._v2.index.Index64.empty(0, self.nplike)
            return ak._v2.contents.numpyarray.NumpyArray(out)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")
        else:
            offsets = ak._v2.index.Index64.zeros(1, self.nplike)
            return (offsets, EmptyArray(None, self._parameters))

    def mergeable(self, other, mergebool):
        if not _parameters_equal(self._parameters, other._parameters):
            return False
        return True

    def mergemany(self, others):
        if len(others) == 0:
            return self

        elif len(others) == 1:
            return others[0]

        else:
            tail_others = others[1:]
            return others[0].mergemany(tail_others)

    def fillna(self, value):
        return EmptyArray(None, self._parameters)

    def _localindex(self, axis, depth):
        return ak._v2.contents.numpyarray.NumpyArray(np.empty(0, np.int64))

    def numbers_to_type(self, name):
        return ak._v2.contents.emptyarray.EmptyArray(self._identifier, self._parameters)

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
        as_numpy = ak._v2.contents.NumpyArray(self)
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
        return ak._v2.contents.emptyarray.EmptyArray(self._identifier, self._parameters)

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
    ):
        as_numpy = self.toNumpyArray(reducer.preferred_dtype, nplike=parents.nplike)
        return as_numpy._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            mask,
            keepdims,
        )

    def _validityerror(self, path):
        return ""

    def _nbytes_part(self):
        if self.identifier is not None:
            return self.identifier._nbytes_part()
        return 0

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis != depth:
            raise np.AxisError(
                "axis={0} exceeds the depth of this array({1})".format(axis, depth)
            )
        else:
            return self.rpad_axis0(target, True)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if options["emptyarray_to"] is None:
            return pyarrow.Array.from_buffers(
                ak._v2._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.null(), options["extensionarray"], mask_node, self
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
                nplike=numpy,
            )
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

    def _to_numpy(self, allow_missing):
        return self.nplike.empty(0, dtype=np.float64)

    def _completely_flatten(self, nplike, options):
        return []

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return EmptyArray(self._identifier, None)

        else:

            def continuation():
                pass

        result = action(
            self,
            depth=depth,
            depth_context=depth_context,
            lateral_context=lateral_context,
            continuation=continuation,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def packed(self):
        return self

    def _to_list(self, behavior):
        return []
