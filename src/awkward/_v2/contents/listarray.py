# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, NestedIndexError
from awkward._v2.forms.listform import ListForm

np = ak.nplike.NumpyMetadata.instance()


class ListArray(Content):
    def __init__(self, starts, stops, content, identifier=None, parameters=None):
        if not isinstance(starts, Index) and starts.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                "{0} 'starts' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(starts))
            )
        if not (isinstance(stops, Index) and starts.dtype == stops.dtype):
            raise TypeError(
                "{0} 'stops' must be an Index with the same dtype as 'starts' ({1}), "
                "not {2}".format(type(self).__name__, repr(starts.dtype), repr(stops))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not len(starts) <= len(stops):
            raise ValueError(
                "{0} len(starts) ({1}) must be <= len(stops) ({2})".format(
                    type(self).__name__, len(starts), len(stops)
                )
            )

        self._starts = starts
        self._stops = stops
        self._content = content
        self._init(identifier, parameters)

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._starts.nplike

    Form = ListForm

    @property
    def form(self):
        return self.Form(
            self._starts.form,
            self._stops.form,
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return len(self._starts)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListArray len="]
        out.append(repr(str(len(self))))
        out.append(">\n")
        out.append(self._starts._repr(indent + "    ", "<starts>", "</starts>\n"))
        out.append(self._stops._repr(indent + "    ", "<stops>", "</stops>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ListArray>")
        out.append(post)
        return "".join(out)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
            raise NestedIndexError(self, where)
        start, stop = self._starts[where], self._stops[where]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return ListArray(
            self._starts[start:stop],
            self._stops[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextstarts = self._starts[carry.data]
            nextstops = self._stops[: len(self._starts)][carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return ListArray(
            nextstarts,
            nextstops,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike  # noqa: F841

        if head == ():
            return self

        elif isinstance(head, int):
            raise NotImplementedError

        elif isinstance(head, slice):
            raise NotImplementedError

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NotImplementedError

        else:
            raise AssertionError(repr(head))
