# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.indexedoptionform import IndexedOptionForm

np = ak.nplike.NumpyMetadata.instance()


class IndexedOptionArray(Content):
    def __init__(self, index, content, identifier=None, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.int64),
            )
        ):
            raise TypeError(
                "{0} 'index' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(index))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._index = index
        self._content = content
        self._init(identifier, parameters)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._index.nplike

    @property
    def nonvirtual_nplike(self):
        return self._index.nplike

    Form = IndexedOptionForm

    @property
    def form(self):
        return self.Form(
            self._index.form,
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return len(self._index)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedOptionArray len="]
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedOptionArray>")
        out.append(post)
        return "".join(out)

    def toIndexedOptionArray64(self):
        if self._index.dtype == np.dtype(np.int64):
            return self
        else:
            return IndexedOptionArray(
                self._index.astype(np.int64),
                self._content,
                identifier=self._identifier,
                parameters=self._parameters,
            )

    def mask_as_bool(self, valid_when=True):
        if valid_when:
            return self._index.data >= 0
        else:
            return self._index.data < 0

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
            raise NestedIndexError(self, where)
        if self._index[where] < 0:
            return None
        else:
            return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return IndexedOptionArray(
            self._index[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedOptionArray(
            self._index,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedOptionArray(
            self._index,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return IndexedOptionArray(
            nextindex,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _nextcarry_outindex(self, nplike):
        numnull = ak._v2.index.Index64.empty(1, nplike)

        self._handle_error(
            nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.to(nplike),
                self._index.to(nplike),
                len(self._index),
            )
        )
        nextcarry = ak._v2.index.Index64.empty(len(self._index) - numnull[0], nplike)
        outindex = ak._v2.index.Index.empty(len(self._index), nplike, self._index.dtype)

        self._handle_error(
            nplike[
                "awkward_IndexedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.to(nplike),
                outindex.to(nplike),
                self._index.to(nplike),
                len(self._index),
                len(self._content),
            )
        )

        return numnull[0], nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        nplike = self.nplike
        if len(slicestarts) != len(self):
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "cannot fit jagged slice with length {0} into {1} of size {2}".format(
                    len(slicestarts), type(self).__name__, len(self)
                ),
            )

        numnull, nextcarry, outindex = self._nextcarry_outindex(nplike)

        reducedstarts = ak._v2.index.Index64.empty(len(self) - numnull, nplike)
        reducedstops = ak._v2.index.Index64.empty(len(self) - numnull, nplike)
        self._handle_error(
            nplike[
                "awkward_MaskedArray_getitem_next_jagged_project",
                outindex.dtype.type,
                slicestarts.dtype.type,
                slicestops.dtype.type,
                reducedstarts.dtype.type,
                reducedstops.dtype.type,
            ](
                outindex.to(nplike),
                slicestarts.to(nplike),
                slicestops.to(nplike),
                reducedstarts.to(nplike),
                reducedstops.to(nplike),
                len(self),
            )
        )
        next = self._content._carry(nextcarry, True, NestedIndexError)
        out = next._getitem_next_jagged(reducedstarts, reducedstops, slicecontent, tail)
        out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outindex, out, self._identifier, self._parameters
        )
        return out2._simplify_optiontype()

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            numnull, nextcarry, outindex = self._nextcarry_outindex(self.nplike)

            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._getitem_next(head, tail, advanced)
            out2 = IndexedOptionArray(outindex, out, self._identifier, self._parameters)
            return out2._simplify_optiontype()

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list) and ak._util.isstr(head[0]):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def project(self):
        numnull = ak._v2.index.Index64.empty(1, self.nplike)

        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](numnull.to(self.nplike), self._index.to(self.nplike), len(self._index))
        )

        nextcarry = ak._v2.index.Index64.empty(len(self) - numnull[0], self.nplike)

        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_flatten_nextcarry",
                nextcarry.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.to(self.nplike),
                self._index.to(self.nplike),
                len(self._index),
                len(self._content),
            )
        )

        return self._content._carry(nextcarry, False, NestedIndexError)

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self.nplike)

            next = self._content._carry(nextcarry, False, NestedIndexError)
            out = next._localindex(posaxis, depth)
            out2 = ak.v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
            )
            return out2

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self.nplike)
            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
            )
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex, out, parameters=parameters
            )
            return out2._simplify_optiontype()
