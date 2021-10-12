# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.indexedform import IndexedForm

np = ak.nplike.NumpyMetadata.instance()


class IndexedArray(Content):
    def __init__(self, index, content, identifier=None, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.uint32),
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

    Form = IndexedForm

    @property
    def form(self):
        return self.Form(
            self._index.form,
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return IndexedArray(
            ak._v2.index.Index(self._index.to(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._index)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedArray len="]
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedArray>")
        out.append(post)
        return "".join(out)

    def toIndexedOptionArray64(self):
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            self._index, self._content, self._identifier, self._parameters
        )

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return IndexedArray(
            self._index[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedArray(
            self._index,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedArray(
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

        return IndexedArray(
            nextindex,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        nplike = self.nplike
        if len(slicestarts) != len(self) and nplike.known_shape:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "cannot fit jagged slice with length {0} into {1} of size {2}".format(
                    len(slicestarts), type(self).__name__, len(self)
                ),
            )

        nextcarry = ak._v2.index.Index64.zeros(len(self), nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_getitem_nextcarry",
                nextcarry.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.to(nplike),
                self._index.to(nplike),
                len(self._index),
                len(self._content),
            )
        )
        # an eager carry (allow_lazy = false) to avoid infinite loop (unproven)
        next = self._content._carry(nextcarry, False, NestedIndexError)
        return next._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike

        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            nextcarry = ak._v2.index.Index64.empty(len(self._index), nplike)
            self._handle_error(
                nplike[
                    "awkward_IndexedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.to(nplike),
                    self._index.to(nplike),
                    len(self._index),
                    len(self._content),
                )
            )

            next = self._content._carry(nextcarry, False, NestedIndexError)
            return next._getitem_next(head, tail, advanced)

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
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def project(self):
        nextcarry = ak._v2.index.Index64.empty(len(self), self.nplike)
        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_getitem_nextcarry",
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
            return self.project()._localindex(posaxis, depth)

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
        if len(self._index) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        next = self._content._carry(self._index, False, NestedIndexError)
        return next._argsort_next(
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

    def _sort_next(
        self,
        negaxis,
        starts,
        parents,
        outlength,
        ascending,
        stable,
        kind,
        order,
    ):
        next = self._content._carry(self._index, False, NestedIndexError)
        return next._sort_next(
            negaxis,
            starts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            return self.project()._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
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
        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)

        nextcarry = ak._v2.index.Index64.zeros(index_length, nplike)
        nextparents = ak._v2.index.Index64.zeros(index_length, nplike)
        outindex = ak._v2.index.Index64.zeros(index_length, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.to(nplike),
                nextparents.to(nplike),
                outindex.to(nplike),
                self._index.to(nplike),
                parents.to(nplike),
                index_length,
            )
        )
        next = self._content._carry(nextcarry, False, NestedIndexError)
        nextshifts = None
        out = next._reduce_next(
            reducer,
            negaxis,
            starts,
            nextshifts,
            nextparents,
            outlength,
            mask,
            keepdims,
        )

        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)

            elif isinstance(out, ak._v2.contents.ListOffsetArray):
                if len(starts) > 0 and starts[0] != 0:
                    raise AssertionError(
                        "reduce_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero ({0})".format(
                            starts[0]
                        )
                    )

                outoffsets = ak._v2.index.Index64.empty(len(starts) + 1, nplike)
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_fix_offsets_64",
                        outoffsets.dtype.type,
                        starts.dtype.type,
                    ](
                        outoffsets.to(nplike),
                        starts.to(nplike),
                        len(starts),
                        len(self._index),
                    )
                )

                tmp = ak._v2.contents.IndexedArray(
                    outindex,
                    out._content,
                    None,
                    None,
                )

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    None,
                )

            else:
                raise AssertionError(
                    "reduce_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray64; "
                    "instead, it returned " + out.classname
                )

    def _validityerror(self, path):
        error = self.nplike["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.to(self.nplike), len(self.index), len(self.content), False
        )
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")
            message = error.str.decode(errors="surrogateescape")
            return 'at {0} ("{1}"): {2} at i={3}{4}'.format(
                path, type(self), message, error.id, filename
            )

        elif isinstance(
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
