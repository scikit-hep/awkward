# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numpy as np

import awkward as ak
from awkward._v2.contents.content import Content, NestedIndexError


class RegularArray(Content):
    def __init__(self, content, size, zeros_length=0, identifier=None, parameters=None):
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not (ak._util.isint(size) and size >= 0):
            raise TypeError(
                "{0} 'size' must be a non-negative integer, not {1}".format(
                    type(self).__name__, size
                )
            )
        if not (ak._util.isint(zeros_length) and zeros_length >= 0):
            raise TypeError(
                "{0} 'zeros_length' must be a non-negative integer, not {1}".format(
                    type(self).__name__, zeros_length
                )
            )

        self._content = content
        self._size = int(size)
        if size != 0:
            self._length = len(content) // size  # floor division
        else:
            self._length = zeros_length
        self._init(identifier, parameters)

    @property
    def size(self):
        return self._size

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._content.nplike

    @property
    def form(self):
        return ak._v2.forms.RegularForm(
            self._content.form,
            self._size,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RegularArray len="]
        out.append(repr(str(len(self))))
        out.append(" size=")
        out.append(repr(str(self._size)))
        out.append(">\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</RegularArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
            raise NestedIndexError(self, where)
        start, stop = (where) * self._size, (where + 1) * self._size
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        zeros_length = stop - start
        substart, substop = start * self._size, stop * self._size
        return RegularArray(
            self._content._getitem_range(slice(substart, substop)),
            self._size,
            zeros_length,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where):
        return RegularArray(
            self._content._getitem_field(where),
            self._size,
            self._length,
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where):
        return RegularArray(
            self._content._getitem_fields(where),
            self._size,
            self._length,
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        nplike, where = carry.nplike, carry.data

        copied = allow_lazy == "copied"
        if not issubclass(where.dtype.type, np.int64):
            where = where.astype(np.int64)
            copied = True

        negative = where < 0
        if nplike.any(negative):
            if not copied:
                where = where.copy()
                copied = True
            where[negative] += self._length

        if nplike.any(where >= self._length):
            raise NestedIndexError(self, where)

        nextcarry = ak._v2.index.Index64.empty(len(where) * self._size, nplike)
        self._handle_error(
            nplike[
                "awkward_RegularArray_getitem_carry",
                nextcarry.dtype.type,
                where.dtype.type,
            ](
                nextcarry.to(nplike),
                where,
                len(where),
                self._size,
            ),
            carry,
        )

        return RegularArray(
            self._content._carry(nextcarry, allow_lazy, exception),
            self._size,
            len(where),
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike

        if head == ():
            raise NotImplementedError

        elif isinstance(head, int):
            nexthead, nexttail = self._headtail(tail)
            nextcarry = ak._v2.index.Index64.empty(self._length, nplike)
            self._handle_error(
                nplike["awkward_RegularArray_getitem_next_at", nextcarry.dtype.type](
                    nextcarry.to(nplike),
                    head,
                    self._length,
                    self._size,
                ),
                head,
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            print(f"{self._length = } {head = } {tail = }")

            nexthead, nexttail = self._headtail(tail)
            start, stop, step = head.indices(self._size)

            print(f"{start = } {stop = } {step = }")

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

            print(f"{nextsize = }")

            nextcarry = ak._v2.index.Index64.empty(self._length * nextsize, nplike)
            self._handle_error(
                nplike[
                    "awkward_RegularArray_getitem_next_range",
                    nextcarry.dtype.type,
                ](
                    nextcarry.to(nplike),
                    start,
                    step,
                    self._length,
                    self._size,
                    nextsize,
                ),
                head,
            )

            print(f"{nextcarry = }")

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

            if advanced is None or len(advanced) == 0:
                print("not advanced")

                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    nextsize,
                    self._length,
                    self._identifier,
                    self._parameters,
                )
            else:
                print("advanced")

                nextadvanced = ak._v2.index.Index64.empty(self._length * nextsize, nplike)
                self._handle_error(
                    nplike[
                        "awkward_RegularArray_getitem_next_range_spreadadvanced",
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

                print(f"{nextadvanced = }")

                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    nextsize,
                    self._length,
                    self._identities,
                    self._parameters,
                )

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
