# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, NestedIndexError
from awkward._v2.forms.listoffsetform import ListOffsetForm


np = ak.nplike.NumpyMetadata.instance()


class ListOffsetArray(Content):
    def __init__(self, offsets, content, identifier=None, parameters=None):
        if not isinstance(offsets, Index) and offsets.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                "{0} 'offsets' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(offsets))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not len(offsets) >= 1:
            raise ValueError(
                "{0} len(offsets) ({1}) must be >= 1".format(
                    type(self).__name__, len(offsets)
                )
            )

        self._offsets = offsets
        self._content = content
        self._init(identifier, parameters)

    @property
    def starts(self):
        return self._offsets[:-1]

    @property
    def stops(self):
        return self._offsets[1:]

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._offsets.nplike

    Form = ListOffsetForm

    @property
    def form(self):
        return self.Form(
            self._offsets.form,
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return len(self._offsets) - 1

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListOffsetArray len="]
        out.append(repr(str(len(self))))
        out.append(">\n")
        out.append(self._offsets._repr(indent + "    ", "<offsets>", "</offsets>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ListOffsetArray>")
        out.append(post)
        return "".join(out)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
            raise NestedIndexError(self, where)
        start, stop = self._offsets[where], self._offsets[where + 1]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        offsets = self._offsets[start : stop + 1]
        if len(offsets) == 0:
            offsets = Index(self._offsets.nplike.array([0], dtype=self._offsets.dtype))
        return ListOffsetArray(
            offsets,
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextstarts = self.starts[carry.data]
            nextstops = self.stops[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return ak._v2.contents.listarray.ListArray(
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
            nexthead, nexttail = self._headtail(tail)
            nextcarry = ak._v2.index.Index64.empty(self._starts.length, nplike)
            self._handle_error(
                nplike["ListArray_getitem_next_at", nextcarry.dtype.type](
                    nextcarry.to(nplike),
                    self._starts.data,
                    self._stops.data,
                    self._starts.length,
                    head,
                ),
                head,
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = self._headtail(tail)
            lenstarts = len(self._starts)
            start, stop, step = head.indices(lenstarts)

            nextsize = 0
            if step > 0 and stop - start > 0:
                diff = stop - start
                nextsize = diff // step
                if diff % step != 0:
                    nextsize += 1
            elif step < 0 and stop - start < 0:
                diff = start - stop
                nextsize = diff // step
                if diff % step != 0:
                    nextsize += 1

            nextcarry = ak._v2.index.Index64.empty(nextsize * len(self), nplike)
            nextoffsets = ak._v2.index.Index32.empty(lenstarts + 1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    nextoffsets.dtype.type,
                    nextoffsets.dtype.type,
                ](
                    nextoffsets.to(nplike),
                    nextcarry.to(nplike),
                    self.starts.data,
                    self.stops.data,
                    lenstarts,
                    start,
                    stop,
                    step,
                ),
                head,
            )

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

            if advanced is None or len(advanced) == 0:
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    self._identifier,
                    self._parameters,
                )
            else:
                nextadvanced = ak._v2.index.Index64.empty(
                    self._length * nextsize, nplike
                )
                self._handle_error(
                    nplike[
                        "awkward_ListOffsetArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextadvanced.to(nplike),
                        advanced.to(nplike),
                        self._length,
                        nextsize,
                    ),
                    head,
                )
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    self._identifier,
                    self._parameters,
                )

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
