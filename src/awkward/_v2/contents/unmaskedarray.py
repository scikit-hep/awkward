# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.forms.unmaskedform import UnmaskedForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()


class UnmaskedArray(Content):
    is_OptionType = True

    def __init__(self, content, identifier=None, parameters=None, nplike=None):
        if not isinstance(content, Content):
            raise TypeError(
                "{} 'content' must be a Content subtype, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if nplike is None:
            nplike = content.nplike

        self._content = content
        self._init(identifier, parameters, nplike)

    @property
    def content(self):
        return self._content

    Form = UnmaskedForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._content._form_with_key(getkey),
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return UnmaskedArray(
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._content.length

    def _forget_length(self):
        return UnmaskedArray(
            self._content._forget_length(),
            self._identifier,
            self._parameters,
            self._nplike,
        )

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

    def merge_parameters(self, parameters):
        return UnmaskedArray(
            self._content,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toByteMaskedArray(self):
        return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
            ak._v2.index.Index8(self.mask_as_bool(valid_when=True).view(np.int8)),
            self._content,
            True,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def toIndexedOptionArray64(self):
        arange = self._nplike.arange(self._content.length, dtype=np.int64)
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            ak._v2.index.Index64(arange),
            self._content,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def mask_as_bool(self, valid_when=True, nplike=None):
        if nplike is None:
            nplike = self._nplike

        if valid_when:
            return nplike.ones(self._content.length, dtype=np.bool_)
        else:
            return nplike.zeros(self._content.length, dtype=np.bool_)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return ak._v2._typetracer.MaybeNone(self._content._getitem_at(where))

        return self._content._getitem_at(where)

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return UnmaskedArray(
            self._content._getitem_range(slice(start, stop)),
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return UnmaskedArray(
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
            self._nplike,
        )

    def _getitem_fields(self, where, only_fields=()):
        return UnmaskedArray(
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
            self._nplike,
        )

    def _carry(self, carry, allow_lazy, exception):
        return UnmaskedArray(
            self._content._carry(carry, allow_lazy, exception),
            self._carry_identifier(carry, exception),
            self._parameters,
            self._nplike,
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            return UnmaskedArray(
                self._content._getitem_next(head, tail, advanced),
                self._identifier,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NotImplementedError

        else:
            raise AssertionError(repr(head))

    def project(self, mask=None):
        if mask is not None:
            return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
                mask,
                self._content,
                False,
                None,
                self._parameters,
                self._nplike,
            ).project()
        else:
            return self._content

    def simplify_optiontype(self):
        if isinstance(
            self._content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self._content
        else:
            return self

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self._nplike)
            out[0] = self.length
            return ak._v2.contents.numpyarray.NumpyArray(out, None, None, self._nplike)[
                0
            ]
        else:
            return ak._v2.contents.unmaskedarray.UnmaskedArray(
                self._content.num(posaxis, depth), None, self._parameters, self._nplike
            )

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")
        else:
            offsets, flattened = self._content._offsets_and_flattened(posaxis, depth)
            if offsets.length == 0:
                return (
                    offsets,
                    UnmaskedArray(flattened, None, self._parameters, self._nplike),
                )

            else:
                return (offsets, flattened)

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

        if isinstance(
            other,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self._content.mergeable(other.content, mergebool)

        else:
            return self._content.mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.toIndexedOptionArray64()._reverse_merge(other)

    def _mergemany(self, others):
        if len(others) == 0:
            return self
        return self.toIndexedOptionArray64().mergemany(others)

    def fillna(self, value):
        return self._content.fillna(value)

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            return UnmaskedArray(
                self._content._localindex(posaxis, depth),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def numbers_to_type(self, name):
        return ak._v2.contents.unmaskedarray.UnmaskedArray(
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
            self._nplike,
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
        if self._content.length == 0:
            return ak._v2.contents.NumpyArray(
                self._nplike.empty(0, np.int64), None, None, self._nplike
            )

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

        if isinstance(out, ak._v2.contents.RegularArray):
            tmp = ak._v2.contents.UnmaskedArray(
                out._content,
                None,
                None,
                self._nplike,
            ).simplify_optiontype()

            return ak._v2.contents.RegularArray(
                tmp,
                out._size,
                out._length,
                None,
                None,
                self._nplike,
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

        if isinstance(out, ak._v2.contents.RegularArray):
            tmp = ak._v2.contents.UnmaskedArray(
                out._content,
                self._identifier,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()

            return ak._v2.contents.RegularArray(
                tmp,
                out._size,
                out._length,
                self._identifier,
                self._parameters,
                self._nplike,
            )

        else:
            return out

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            return ak._v2.contents.unmaskedarray.UnmaskedArray(
                self._content._combinations(
                    n, replacement, recordlookup, parameters, posaxis, depth
                ),
                self._identifier,
                self._parameters,
                self._nplike,
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
        next = self._content
        if isinstance(next, ak._v2_contents.RegularArray):
            next = next.toListOffsetArray64(True)

        return next._reduce_next(
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
        if isinstance(
            self._content,
            (
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return "{0} contains \"{1}\", the operation that made it might have forgotten to call 'simplify_optiontype()'"
        else:
            return self._content.validityerror(path + ".content")

    def _nbytes_part(self):
        result = self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        elif posaxis == depth + 1:
            return self._content._rpad(target, posaxis, depth, clip)
        else:
            return ak._v2.contents.unmaskedarray.UnmaskedArray(
                self._content._rpad(target, posaxis, depth, clip),
                None,
                self._parameters,
                self._nplike,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self._content._to_arrow(pyarrow, self, None, length, options)

    def _to_numpy(self, allow_missing):
        content = ak._v2.operations.convert.to_numpy(
            self.content, allow_missing=allow_missing
        )
        if allow_missing:
            return self._nplike.ma.MaskedArray(content)
        else:
            return content

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return UnmaskedArray(
                    self._content._recursively_apply(
                        action,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                    self._nplike,
                )

        else:

            def continuation():
                self._content._recursively_apply(
                    action,
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
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def packed(self):
        return UnmaskedArray(
            self._content.packed(), self._identifier, self._parameters, self._nplike
        )

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        return self._content._to_list(behavior)

    def _to_nplike(self, nplike):
        content = self._content._to_nplike(nplike)
        return UnmaskedArray(
            content,
            identifier=self.identifier,
            parameters=self.parameters,
            nplike=nplike,
        )

    def _to_json(
        self,
        nan_string,
        infinity_string,
        minus_infinity_string,
        complex_real_string,
        complex_imag_string,
    ):
        out = self._to_json_custom()
        if out is not None:
            return out

        return self._content._to_json(
            nan_string,
            infinity_string,
            minus_infinity_string,
            complex_real_string,
            complex_imag_string,
        )
