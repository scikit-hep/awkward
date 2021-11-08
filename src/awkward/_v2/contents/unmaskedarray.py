# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.forms.unmaskedform import UnmaskedForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()


class UnmaskedArray(Content):
    is_OptionType = True

    def __init__(self, content, identifier=None, parameters=None):
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._content = content
        self._init(identifier, parameters)

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._content.nplike

    Form = UnmaskedForm

    @property
    def form(self):
        return self.Form(
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        return UnmaskedArray(
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._content)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnmaskedArray len="]
        out.append(repr(str(len(self))))
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
        )

    def toByteMaskedArray(self):
        return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
            ak._v2.index.Index8(self.mask_as_bool(valid_when=True).view(np.int8)),
            self._content,
            valid_when=True,
            identifier=self._identifier,
            parameters=self._parameters,
        )

    def toIndexedOptionArray64(self):
        arange = self._content.nplike.arange(len(self._content), dtype=np.int64)
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            ak._v2.index.Index64(arange),
            self._content,
            self._identifier,
            self._parameters,
        )

    def mask_as_bool(self, valid_when=True, nplike=None):
        if nplike is None:
            nplike = self._content.nplike

        if valid_when:
            return nplike.ones(len(self._content), dtype=np.bool_)
        else:
            return nplike.zeros(len(self._content), dtype=np.bool_)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        return self._content._getitem_at(where)

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return UnmaskedArray(
            self._content._getitem_range(slice(start, stop)),
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return UnmaskedArray(
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return UnmaskedArray(
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        return UnmaskedArray(
            self.content._carry(carry, allow_lazy, exception),
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            return UnmaskedArray(
                self.content._getitem_next(head, tail, advanced),
                self._identifier,
                self._parameters,
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
                self.content,
                valid_when=False,
                identifier=None,
                parameters=self.parameters,
            ).project()
        else:
            return self.content

    def simplify_optiontype(self):
        if isinstance(
            self.content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self.content
        else:
            return self

    def mergeable(self, other, mergebool):
        if not _parameters_equal(self._parameters, other._parameters):
            return False

        if isinstance(
            other,
            (
                ak._v2.contents.emptyArray.EmptyArray,
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
            self.content.mergeable(other.content, mergebool)

        else:
            return self.content.mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.toIndexedOptionArray64()._reverse_merge(other)

    def _mergemany(self, others):
        if len(others) == 0:
            return self
        return self.toIndexedOptionArray64().mergemany(others)

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            return UnmaskedArray(
                self._content._localindex(posaxis, depth),
                self._identifier,
                self._parameters,
            )

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
        if len(self._content) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

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
            ).simplify_optiontype()

            return ak._v2.contents.RegularArray(
                tmp,
                out._size,
                out._length,
                None,
                None,
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
            ).simplify_optiontype()

            return ak._v2.contents.RegularArray(
                tmp,
                out._size,
                out._length,
                self._identifier,
                self._parameters,
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
            self.content,
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
            return self.content.validityerror(path + ".content")

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
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self._content._to_arrow(pyarrow, self, None, length, options)

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
