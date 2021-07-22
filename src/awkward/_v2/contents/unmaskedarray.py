# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content, NestedIndexError  # noqa: F401

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
    def form(self):
        return ak._v2.forms.UnmaskedForm(
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
        out.append(">\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</UnmaskedArray>")
        out.append(post)
        return "".join(out)

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

    def _getitem_field(self, where):
        return UnmaskedArray(
            self._content[where],
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where):
        return UnmaskedArray(
            self._content[where],
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
        nplike = self.nplike  # noqa: F841

        if head is None:
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
