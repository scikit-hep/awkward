# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.unmaskedform import UnmaskedForm
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class UnmaskedArray(Content):
    is_option = True

    def __init__(self, content, *, parameters=None):
        if not isinstance(content, Content):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'content' must be a Content subtype, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if content.is_union or content.is_indexed or content.is_option:
            raise ak._errors.wrap_error(
                TypeError(
                    "{0} cannot contain a union-type, option-type, or indexed 'content' ({1}); try {0}.simplified instead".format(
                        type(self).__name__, type(content).__name__
                    )
                )
            )
        self._content = content
        self._init(parameters, content.backend)

    @property
    def content(self):
        return self._content

    form_cls: Final = UnmaskedForm

    def copy(self, content=unset, *, parameters=unset):
        return UnmaskedArray(
            self._content if content is unset else content,
            parameters=self._parameters if parameters is unset else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, content, *, parameters=None):
        if content.is_union:
            return content.copy(
                contents=[cls.simplified(x) for x in content.contents],
                parameters=ak._util.merge_parameters(content._parameters, parameters),
            )
        elif content.is_indexed or content.is_option:
            return content.copy(
                parameters=ak._util.merge_parameters(content._parameters, parameters)
            )
        else:
            return cls(content, parameters=parameters)

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        self._content._to_buffers(form.content, getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        return UnmaskedArray(
            self._content._to_typetracer(forget_length),
            parameters=self._parameters,
        )

    def _touch_data(self, recursive):
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive):
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self):
        return self._content.length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnmaskedArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</UnmaskedArray>")
        out.append(post)
        return "".join(out)

    def to_IndexedOptionArray64(self):
        arange = self._backend.index_nplike.arange(self._content.length, dtype=np.int64)
        return ak.contents.IndexedOptionArray(
            ak.index.Index64(arange, nplike=self._backend.index_nplike),
            self._content,
            parameters=self._parameters,
        )

    def to_ByteMaskedArray(self, valid_when):
        return ak.contents.ByteMaskedArray(
            ak.index.Index8(
                self.mask_as_bool(valid_when).view(np.int8),
                nplike=self._backend.index_nplike,
            ),
            self._content,
            valid_when,
            parameters=self._parameters,
        )

    def to_BitMaskedArray(self, valid_when, lsb_order):
        bitlength = int(numpy.ceil(self._content.length / 8.0))
        if valid_when:
            bitmask = self._backend.index_nplike.full(
                bitlength, np.uint8(255), dtype=np.uint8
            )
        else:
            bitmask = self._backend.index_nplike.zeros(bitlength, dtype=np.uint8)

        return ak.contents.BitMaskedArray(
            ak.index.IndexU8(bitmask),
            self._content,
            valid_when,
            self.length,
            lsb_order,
            parameters=self._parameters,
        )

    def mask_as_bool(self, valid_when=True):
        if valid_when:
            return self._backend.index_nplike.ones(self._content.length, dtype=np.bool_)
        else:
            return self._backend.index_nplike.zeros(
                self._content.length, dtype=np.bool_
            )

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return ak._typetracer.MaybeNone(self._content._getitem_at(where))

        return self._content._getitem_at(where)

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return UnmaskedArray(
            self._content._getitem_range(slice(start, stop)),
            parameters=self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return UnmaskedArray.simplified(
            self._content._getitem_field(where, only_fields), parameters=None
        )

    def _getitem_fields(self, where, only_fields=()):
        return UnmaskedArray.simplified(
            self._content._getitem_fields(where, only_fields), parameters=None
        )

    def _carry(self, carry, allow_lazy):
        return UnmaskedArray.simplified(
            self._content._carry(carry, allow_lazy), parameters=self._parameters
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return UnmaskedArray(
            self._content._getitem_next_jagged(
                slicestarts, slicestops, slicecontent, tail
            ),
            parameters=self._parameters,
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(
            head, (int, slice, ak.index.Index64, ak.contents.ListOffsetArray)
        ):
            return UnmaskedArray.simplified(
                self._content._getitem_next(head, tail, advanced),
                parameters=self._parameters,
            )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def project(self, mask=None):
        if mask is not None:
            return ak.contents.ByteMaskedArray(
                mask, self._content, False, parameters=self._parameters
            ).project()
        else:
            return self._content

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(np.AxisError("axis=0 not allowed for flatten"))
        else:
            offsets, flattened = self._content._offsets_and_flattened(axis, depth)
            if offsets.length == 0:
                return (
                    offsets,
                    UnmaskedArray(flattened, parameters=self._parameters),
                )

            else:
                return (offsets, flattened)

    def _mergeable_next(self, other, mergebool):
        if isinstance(
            other,
            (
                ak.contents.IndexedArray,
                ak.contents.IndexedOptionArray,
                ak.contents.ByteMaskedArray,
                ak.contents.BitMaskedArray,
                ak.contents.UnmaskedArray,
            ),
        ):
            return self._content._mergeable(other.content, mergebool)

        else:
            return self._content._mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.to_IndexedOptionArray64()._reverse_merge(other)

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        if all(isinstance(x, UnmaskedArray) for x in others):
            parameters = self._parameters
            tail_contents = []
            for x in others:
                parameters = ak._util.merge_parameters(parameters, x._parameters, True)
                tail_contents.append(x._content)

            return UnmaskedArray(
                self._content._mergemany(tail_contents), parameters=parameters
            )

        else:
            return self.to_IndexedOptionArray64()._mergemany(others)

    def _fill_none(self, value: Content) -> Content:
        return self._content._fill_none(value)

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            return UnmaskedArray(
                self._content._local_index(axis, depth), parameters=self._parameters
            )

    def _numbers_to_type(self, name):
        return ak.contents.UnmaskedArray(
            self._content._numbers_to_type(name), parameters=self._parameters
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._content.length == 0:
            return True
        return self._content._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        if self._content.length == 0:
            return self
        return self._content._unique(negaxis, starts, parents, outlength)

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
        out = self._content._argsort_next(
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

        if isinstance(out, ak.contents.RegularArray):
            tmp = ak.contents.UnmaskedArray.simplified(out._content, parameters=None)
            return ak.contents.RegularArray(
                tmp, out._size, out._length, parameters=None
            )

        else:
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        out = self._content._sort_next(
            negaxis,
            starts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

        if isinstance(out, ak.contents.RegularArray):
            tmp = ak.contents.UnmaskedArray.simplified(
                out._content, parameters=self._parameters
            )

            return ak.contents.RegularArray(
                tmp, out._size, out._length, parameters=self._parameters
            )

        else:
            return out

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            return ak.contents.UnmaskedArray(
                self._content._combinations(
                    n, replacement, recordlookup, parameters, axis, depth
                ),
                parameters=self._parameters,
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
        return self._content._reduce_next(
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
        return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            return self._content._pad_none(target, axis, depth, clip)
        else:
            return ak.contents.UnmaskedArray(
                self._content._pad_none(target, axis, depth, clip),
                parameters=self._parameters,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self._content._to_arrow(pyarrow, self, None, length, options)

    def _to_numpy(self, allow_missing):
        content = self.content._to_numpy(allow_missing)
        if allow_missing:
            return self._backend.nplike.ma.MaskedArray(content)
        else:
            return content

    def _completely_flatten(self, backend, options):
        branch, depth = self.branch_depth
        if branch or options["drop_nones"] or depth > 1:
            return self.project()._completely_flatten(backend, options)
        else:
            return [self]

    def _drop_none(self):
        return self.to_ByteMaskedArray(True)._drop_none()

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:
            if options["return_simplified"]:
                make = UnmaskedArray.simplified
            else:
                make = UnmaskedArray

            def continuation():
                return make(
                    self._content._recursively_apply(
                        action,
                        behavior,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    parameters=self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                self._content._recursively_apply(
                    action,
                    behavior,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )

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
        return UnmaskedArray(self._content.to_packed(), parameters=self._parameters)

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        return self._content._to_list(behavior, json_conversions)

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        content = self._content.to_backend(backend)
        return UnmaskedArray(content, parameters=self._parameters)

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return self.content.is_equal_to(other.content, index_dtype, numpyarray)
