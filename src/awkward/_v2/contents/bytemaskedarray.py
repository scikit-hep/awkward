# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content, NestedIndexError
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class ByteMaskedArray(Content):
    def __init__(self, mask, content, valid_when, identifier=None, parameters=None):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.int8)):
            raise TypeError(
                "{0} 'mask' must be an Index with dtype=int8, not {1}".format(
                    type(self).__name__, repr(mask)
                )
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{0} 'valid_when' must be boolean, not {1}".format(
                    type(self).__name__, repr(valid_when)
                )
            )
        if not len(mask) <= len(content):
            raise ValueError(
                "{0} len(mask) ({1}) must be <= len(content) ({2})".format(
                    type(self).__name__, len(mask), len(content)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(identifier, parameters)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def nplike(self):
        return self._mask.nplike

    @property
    def form(self):
        return ak._v2.forms.ByteMaskedForm(
            self._mask.form,
            self._content.form,
            self._valid_when,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return len(self._mask)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ByteMaskedArray len="]
        out.append(repr(str(len(self))))
        out.append(" valid_when=")
        out.append(repr(str(self._valid_when)))
        out.append(">\n")
        out.append(self._mask._repr(indent + "    ", "<mask>", "</mask>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ByteMaskedArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
            raise NestedIndexError(self, where)
        if self._mask[where] == self._valid_when:
            return self._content._getitem_at(where)
        else:
            return None

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return ByteMaskedArray(
            self._mask[start:stop],
            self._content._getitem_range(slice(start, stop)),
            self._valid_when,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where):
        return ByteMaskedArray(
            self._mask,
            self._content._getitem_field(where),
            self._valid_when,
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where):
        return ByteMaskedArray(
            self._mask,
            self._content._getitem_fields(where),
            self._valid_when,
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextmask = self._mask[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return ByteMaskedArray(
            nextmask,
            self._content._carry(carry, allow_lazy, exception),
            self._valid_when,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike  # noqa: F841

        if head == ():
            raise NotImplementedError

        elif isinstance(head, int):
            raise NotImplementedError

        elif isinstance(head, slice):
            raise NotImplementedError

        elif ak._util.isstr(head):
            raise NotImplementedError

        elif isinstance(head, list):
            raise NotImplementedError

        elif head is np.newaxis:
            raise NotImplementedError

        elif head is Ellipsis:
            raise NotImplementedError

        elif isinstance(head, ak._v2.index.Index64):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NotImplementedError

        else:
            raise AssertionError(repr(head))
