# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.forms.unmaskedform import UnmaskedForm

np = ak.nplike.NumpyMetadata.instance()


class UnmaskedArray(Content):
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

    @property
    def nonvirtual_nplike(self):
        return self._content.nonvirtual_nplike

    Form = UnmaskedForm

    @property
    def form(self):
        return self.Form(
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
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

    def mask_as_bool(self, valid_when=True):
        if valid_when:
            return self._content.nplike.ones(len(self._content), dtype=np.bool_)
        else:
            return self._content.nplike.zeros(len(self._content), dtype=np.bool_)

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
            )._simplify_optiontype()

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

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            return UnmaskedArray(
                self._content._localindex(posaxis, depth),
                self._identifier,
                self._parameters,
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
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
