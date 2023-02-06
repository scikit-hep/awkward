# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward._errors import deprecate
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.emptyform import EmptyForm
from awkward.index import Index
from awkward.typing import TYPE_CHECKING, Final, Self, SupportsIndex, final

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class EmptyArray(Content):
    is_unknown = True
    is_leaf = True

    def __init__(self, *, parameters=None, backend=None):
        if not (parameters is None or len(parameters) == 0):
            deprecate(
                f"{type(self).__name__} cannot contain parameters", version="2.2.0"
            )
        if backend is None:
            backend = ak._backends.NumpyBackend.instance()
        self._init(parameters, backend)

    form_cls: Final = EmptyForm

    def copy(
        self,
        *,
        parameters=unset,
        backend=unset,
    ):
        if not (parameters is unset or parameters is None or len(parameters) == 0):
            deprecate(
                f"{type(self).__name__} cannot contain parameters", version="2.2.0"
            )
        return EmptyArray(
            parameters=self._parameters if parameters is unset else parameters,
            backend=self._backend if backend is unset else backend,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(parameters=copy.deepcopy(self._parameters, memo))

    @classmethod
    def simplified(cls, *, parameters=None, backend=None):
        if not (parameters is None or len(parameters) == 0):
            deprecate(f"{cls.__name__} cannot contain parameters", version="2.2.0")
        return cls(parameters=parameters, backend=backend)

    def _form_with_key(self, getkey):
        return self.form_cls(parameters=self._parameters, form_key=getkey(self))

    def _to_buffers(self, form, getkey, container, backend, byteorder):
        assert isinstance(form, self.form_cls)

    def _to_typetracer(self, forget_length: bool) -> Self:
        return EmptyArray(
            parameters=self._parameters,
            backend=ak._backends.TypeTracerBackend.instance(),
        )

    def _touch_data(self, recursive):
        pass

    def _touch_shape(self, recursive):
        pass

    @property
    def length(self):
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

    def to_NumpyArray(self, dtype, backend=None):
        backend = backend or self._backend
        return ak.contents.NumpyArray(
            backend.nplike.empty(0, dtype=dtype),
            parameters=self._parameters,
            backend=backend,
        )

    def __array__(self, **kwargs):
        return numpy.empty(0, dtype=np.float64)

    def __iter__(self):
        return iter([])

    def _getitem_nothing(self):
        return self

    def _getitem_at(self, where: SupportsIndex):
        raise ak._errors.index_error(self, where, "array is empty")

    def _getitem_range(self, where):
        return self

    def _getitem_field(self, where, only_fields=()):
        raise ak._errors.index_error(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise ak._errors.index_error(self, where, "not an array of records")

    def _carry(self, carry: Index, allow_lazy: bool) -> EmptyArray:
        assert isinstance(carry, ak.index.Index)

        if not carry.nplike.known_shape or carry.length == 0:
            return self
        else:
            raise ak._errors.index_error(self, carry.data, "array is empty")

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        raise ak._errors.index_error(
            self,
            ak.contents.ListArray(
                slicestarts, slicestops, slicecontent, parameters=None
            ),
            "too many jagged slice dimensions for array",
        )

    def _getitem_next(
        self,
        head: SliceItem | tuple,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ) -> Content:
        if head == ():
            return self

        elif isinstance(head, int):
            raise ak._errors.index_error(self, head, "array is empty")

        elif isinstance(head, slice):
            raise ak._errors.index_error(self, head, "array is empty")

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.index.Index64):
            if not head.nplike.known_shape or head.length == 0:
                return self
            else:
                raise ak._errors.index_error(self, head.data, "array is empty")

        elif isinstance(head, ak.contents.ListOffsetArray):
            raise ak._errors.index_error(self, head, "array is empty")

        elif isinstance(head, ak.contents.IndexedOptionArray):
            raise ak._errors.index_error(self, head, "array is empty")

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(
                np.AxisError(self, "axis=0 not allowed for flatten")
            )
        else:
            offsets = ak.index.Index64.zeros(1, nplike=self._backend.index_nplike)
            return (
                offsets,
                EmptyArray(parameters=self._parameters, backend=self._backend),
            )

    def _mergeable_next(self, other, mergebool):
        return True

    def _mergemany(self, others):
        if len(others) == 0:
            return self
        elif len(others) == 1:
            return others[0]
        else:
            return others[0]._mergemany(others[1:])

    def _fill_none(self, value: Content) -> Content:
        return EmptyArray(parameters=self._parameters, backend=self._backend)

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return ak.contents.NumpyArray(
                self._backend.nplike.empty(0, dtype=np.int64),
                parameters=None,
                backend=self._backend,
            )
        else:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )

    def _numbers_to_type(self, name, including_unknown):
        if including_unknown:
            return self.to_NumpyArray(ak.types.numpytype.primitive_to_dtype(name))
        else:
            return self

    def _is_unique(self, negaxis, starts, parents, outlength):
        return True

    def _unique(self, negaxis, starts, parents, outlength):
        return self

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        as_numpy = self.to_NumpyArray(np.float64)
        return as_numpy._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        return self

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return ak.contents.EmptyArray(
            parameters=self._parameters, backend=self._backend
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
        as_numpy = self.to_NumpyArray(reducer.preferred_dtype)
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
        return 0

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 != depth:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        else:
            return self._pad_none_axis0(target, True)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if options["emptyarray_to"] is None:
            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.null(),
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                ],
                null_count=length,
            )

        else:
            dtype = np.dtype(options["emptyarray_to"])
            next = ak.contents.NumpyArray(
                numpy.empty(length, dtype=dtype),
                parameters=self._parameters,
                backend=self._backend,
            )
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

    def _to_backend_array(self, allow_missing, backend):
        return backend.nplike.empty(0, dtype=np.float64)

    def _remove_structure(self, backend, options):
        return []

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return EmptyArray(parameters=None, backend=self._backend)

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
            backend=self._backend,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._errors.wrap_error(AssertionError(result))

    def to_packed(self) -> Self:
        return self

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise ak._errors.wrap_error(
                TypeError("cannot convert typetracer arrays to Python lists")
            )
        return []

    def _to_backend(self, backend: ak._backends.Backend) -> Self:
        return EmptyArray(parameters=self._parameters, backend=backend)

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return True
