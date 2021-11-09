# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.numpyform import NumpyForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class NumpyArray(Content):
    is_NumpyType = True

    def __init__(self, data, identifier=None, parameters=None, nplike=None):
        self._nplike = ak.nplike.of(data) if nplike is None else nplike
        self._data = self._nplike.asarray(data)

        if (
            self._data.dtype not in ak._v2.types.numpytype._dtype_to_primitive
            and not issubclass(self._data.dtype.type, (np.datetime64, np.timedelta64))
        ):
            raise TypeError(
                "{0} 'data' dtype {1} is not supported; must be one of {2}".format(
                    type(self).__name__,
                    repr(self._data.dtype),
                    ", ".join(
                        repr(x) for x in ak._v2.types.numpytype._dtype_to_primitive
                    ),
                )
            )
        if len(self._data.shape) == 0:
            raise TypeError(
                "{0} 'data' must be an array, not {1}".format(
                    type(self).__name__, repr(data)
                )
            )

        self._init(identifier, parameters)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def inner_shape(self):
        return self._data.shape[1:]

    @property
    def strides(self):
        return self._data.strides

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def nplike(self):
        return self._nplike

    @property
    def ptr(self):
        return self._data.ctypes.data

    Form = NumpyForm

    @property
    def form(self):
        return self.Form(
            ak._v2.types.numpytype._dtype_to_primitive[self._data.dtype],
            self._data.shape[1:],
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        return NumpyArray(
            self.to(ak._v2._typetracer.TypeTracer.instance()),
            self._typetracer_identifier(),
            self._parameters,
            nplike=ak._v2._typetracer.TypeTracer.instance(),
        )

    def __len__(self):
        return len(self._data)

    def to(self, nplike):
        return nplike.asarray(self._data)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray dtype="]
        out.append(repr(str(self.dtype)))
        if len(self._data.shape) == 1:
            out.append(" len=" + repr(str(self._data.shape[0])))
        else:
            out.append(
                " shape='({0})'".format(", ".join(str(x) for x in self._data.shape))
            )

        extra = self._repr_extra(indent + "    ")
        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(extra) != 0 or len(arraystr_lines) > 1:
            arraystr_lines = self._nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.extend(extra)
            out.append("\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</NumpyArray>")
        else:
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</NumpyArray>")

        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return NumpyArray(
            self._data,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toRegularArray(self):
        if len(self._data.shape) == 1:
            return self
        else:
            return ak._v2.contents.RegularArray(
                NumpyArray(
                    self._data.reshape((-1,) + self._data.shape[2:]),
                    None,
                    None,
                    nplike=self._nplike,
                ).toRegularArray(),
                self._data.shape[1],
                self._data.shape[0],
                self._identifier,
                self._parameters,
            )

    def maybe_to_nplike(self, nplike):
        return nplike.asarray(self._data)

    def _getitem_nothing(self):
        tmp = self._data[0:0]
        return NumpyArray(
            tmp.reshape((0,) + tmp.shape[2:]),
            self._range_identifier(0, 0),
            None,
            nplike=self._nplike,
        )

    def _getitem_at(self, where):
        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, None, None, nplike=self._nplike)
        else:
            return out

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1

        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        return NumpyArray(
            out,
            self._range_identifier(start, stop),
            self._parameters,
            nplike=self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)
        try:
            nextdata = self._data[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return NumpyArray(
            nextdata,
            self._carry_identifier(carry, exception),
            self._parameters,
            nplike=self._nplike,
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        if self._data.ndim == 1:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "too many jagged slice dimensions for array",
            )
        else:
            next = self.toRegularArray()
            return next._getitem_next_jagged(
                slicestarts, slicestops, slicecontent, tail
            )

    def _getitem_next(self, head, tail, advanced):
        nplike = self._nplike

        if head == ():
            return self

        elif isinstance(head, int):
            where = (slice(None), head) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            if hasattr(out, "shape") and len(out.shape) != 0:
                return NumpyArray(out, None, None, nplike=nplike)
            else:
                return out

        elif isinstance(head, slice) or head is np.newaxis or head is Ellipsis:
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self.parameters, nplike=nplike)
            return out2

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            if advanced is None:
                where = (slice(None), head.data) + tail
            else:
                where = (nplike.asarray(advanced.data), head.data) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            return NumpyArray(out, None, self.parameters, nplike=nplike)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self.parameters, nplike=nplike)
            return out2

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            next = self.toRegularArray()
            return next._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def mergeable(self, other, mergebool):
        if not _parameters_equal(self._parameters, other._parameters):
            return False

        if isinstance(
            other,
            (
                ak._v2.contents.emptyarray.EmptyArray,
                ak._v2.contents.unionarray.UnionArray,
            ),
        ):
            return True

        elif isinstance(
            other,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self.mergeable(other.content, mergebool)

        if self._data.ndim == 0:
            return False

        if isinstance(other, ak._v2.contents.numpyarray.NumpyArray):
            if self._data.ndim != other.data.ndim:
                return False

            if self.dtype != other.dtype and (
                self.dtype == np.datetime64 or other.dtype == np.datetime64
            ):
                return False

            if self.dtype != other.dtype and (
                self.dtype == np.timedelta64 or other.dtype == np.timedelta64
            ):
                return False

            if len(self.shape) > 1 and len(self.shape) != (other.shape):
                return False

            return True
        else:
            return False

    def mergemany(self, others):
        if len(others) == 0:
            return self
        head, tail = self._merging_strategy(others)

        contiguous_arrays = []

        for array in head:
            parameters = dict(self.parameters.items() & array.parameters.items())
            if isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass
            elif isinstance(array, ak._v2.contents.numpyarray.NumpyArray):
                contiguous_arrays.append(array.data)
            else:
                raise AssertionError(
                    "cannot merge "
                    + type(self).__name__
                    + " with "
                    + type(array).__name__
                )

        contiguous_arrays = self.nplike.concatenate(contiguous_arrays)

        next = NumpyArray(contiguous_arrays, self.identifier, parameters)

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        elif len(self.shape) <= 1:
            raise np.AxisError(self, "'axis' out of range for localindex")
        else:
            return self.toRegularArray()._localindex(posaxis, depth)

    def contiguous(self):
        if self._data.ndim >= 1:
            return ak._v2.contents.NumpyArray(self.nplike.ascontiguousarray(self))
        else:
            return self

    def iscontiguous(self):
        x = self._data.dtype.itemsize

        for i in range(len(self.shape), 0, -1):  # FIXME: more Pythonic way to do it?
            if x != self.strides[i - 1]:
                return False
            else:
                x = x * self.shape[i - 1]

        return True

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
        if self.shape[0] == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        if len(self.shape) == 0:
            raise TypeError(
                "{0} attempting to argsort a scalar ".format(type(self).__name__)
            )
        elif len(self.shape) != 1 or not self.iscontiguous():
            contiguous_self = self if self.iscontiguous() else self.contiguous()
            return contiguous_self.toRegularArray()._argsort_next(
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

        else:
            nplike = self.nplike

            parents_length = len(parents)
            offsets_length = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.to(nplike),
                    parents.to(nplike),
                    parents_length,
                )
            )
            offsets_length = offsets_length[0]

            offsets = ak._v2.index.Index64.empty(offsets_length, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.to(nplike),
                    offsets_length,
                    parents.to(nplike),
                    parents_length,
                )
            )

            nextcarry = ak._v2.index.Index64.empty(self.__len__(), nplike)
            self._handle_error(
                nplike[
                    "awkward_argsort",
                    nextcarry.dtype.type,
                    self._data.dtype.type,
                    offsets.dtype.type,
                ](
                    nextcarry.to(nplike),
                    self._data,
                    self.__len__(),
                    offsets.to(nplike),
                    offsets_length,
                    ascending,
                    stable,
                )
            )

            if shifts is not None:
                self._handle_error(
                    nplike[
                        "awkward_NumpyArray_rearrange_shifted",
                        nextcarry.dtype.type,
                        shifts.dtype.type,
                        offsets.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        shifts.to(nplike),
                        len(shifts),
                        offsets.to(nplike),
                        offsets_length,
                        parents.to(nplike),
                        parents_length,
                        starts.to(nplike),
                        len(starts),
                    )
                )
            out = NumpyArray(nextcarry)
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if len(self.shape) == 0:
            raise TypeError(
                "{0} attempting to sort a scalar ".format(type(self).__name__)
            )

        elif len(self.shape) != 1 or not self.iscontiguous():
            contiguous_self = self if self.iscontiguous() else self.contiguous()
            return contiguous_self.toRegularArray()._sort_next(
                negaxis,
                starts,
                parents,
                outlength,
                ascending,
                stable,
                kind,
                order,
            )

        else:
            nplike = self.nplike

            parents_length = len(parents)
            offsets_length = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.to(nplike),
                    parents.to(nplike),
                    parents_length,
                )
            )

            offsets = ak._v2.index.Index64.zeros(offsets_length[0], nplike)

            self._handle_error(
                nplike[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.to(nplike),
                    offsets_length[0],
                    parents.to(nplike),
                    parents_length,
                )
            )

            out = ak._v2.contents.NumpyArray(nplike.empty(len(self._data), self.dtype))
            self._handle_error(
                nplike[
                    "awkward_sort",
                    out._data.dtype.type,
                    self._data.dtype.type,
                    offsets.dtype.type,
                ](
                    out._data,
                    self._data,
                    self.shape[0],
                    offsets.to(nplike),
                    offsets_length[0],
                    parents_length,
                    ascending,
                    stable,
                )
            )
            return out

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif len(self.shape) <= 1:
            raise np.AxisError("'axis' out of range for combinations")
        else:
            return self.toRegularArray()._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
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
    ):
        nplike = self.nplike

        out = reducer.apply(self, parents, outlength)

        if reducer.needs_position:
            if shifts is None:
                self._handle_error(
                    nplike[
                        "awkward_NumpyArray_reduce_adjust_starts_64",
                        out.data.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        out.data,
                        outlength,
                        parents.to(nplike),
                        starts.to(nplike),
                    )
                )
            else:
                self._handle_error(
                    nplike[
                        "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
                        out.data.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                        shifts.dtype.type,
                    ](
                        out.data,
                        outlength,
                        parents.to(nplike),
                        starts.to(nplike),
                        shifts.to(nplike),
                    )
                )

        if mask:
            outmask = ak._v2.index.Index8.zeros(outlength, nplike)
            self._handle_error(
                nplike[
                    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
                    outmask.dtype.type,
                    parents.dtype.type,
                ](
                    outmask.to(nplike),
                    parents.to(nplike),
                    len(parents),
                    outlength,
                )
            )

            out = ak._v2.contents.ByteMaskedArray(
                outmask,
                out,
                False,
                None,
                None,
            )

        if keepdims:
            out = ak._v2.contents.RegularArray(
                out,
                1,
                len(self),
                None,
                None,
            )

        return out

    def _validityerror(self, path):
        if len(self.shape) == 0:
            return 'at {0} ("{1}"): shape is zero-dimensional'.format(path, type(self))
        for i in range(len(self.shape)):
            if self.shape[i] < 0:
                return 'at {0} ("{1}"): shape[{2}] < 0'.format(path, type(self), i)
        for i in range(len(self.strides)):
            if self.strides[i] % self.dtype.itemsize != 0:
                return 'at {0} ("{1}"): shape[{2}] % itemsize != 0'.format(
                    path, type(self), i
                )
        return ""

    def _rpad(self, target, axis, depth, clip):
        if len(self.shape) == 0:
            raise ValueError("cannot rpad a scalar")
        elif len(self.shape) > 1 or not self.iscontiguous():
            return self.toRegularArray()._rpad(target, axis, depth, clip)
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis != depth:
            raise np.AxisError(
                "axis={0} exceeds the depth of this array({1})".format(axis, depth)
            )
        if not clip:
            if target < len(self):
                return self
            else:
                return self._rpad(target, posaxis, depth, clip=True)
        else:
            return self.rpad_axis0(target, clip=True)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if self._data.ndim != 1:
            return self.toRegularArray()._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

        nparray = self.to(numpy)
        storage_type = pyarrow.from_numpy_dtype(nparray.dtype)

        if issubclass(nparray.dtype.type, (bool, np.bool_)):
            nparray = ak._v2._connect.pyarrow.packbits(nparray)

        return pyarrow.Array.from_buffers(
            ak._v2._connect.pyarrow.to_awkwardarrow_type(
                storage_type, options["extensionarray"], mask_node, self
            ),
            length,
            [
                ak._v2._connect.pyarrow.to_validbits(validbytes),
                ak._v2._connect.pyarrow.to_length(nparray, length),
            ],
            null_count=ak._v2._connect.pyarrow.to_null_count(
                validbytes, options["count_nulls"]
            ),
        )

    def _completely_flatten(self, nplike, options):
        return [self.to(nplike).reshape(-1)]

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if self._data.ndim != 1 and options["numpy_to_regular"]:
            return self.toRegularArray()._recursively_apply(
                action, depth, depth_context, lateral_context, options
            )

        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return NumpyArray(self._data, self._identifier, None, self._nplike)

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
