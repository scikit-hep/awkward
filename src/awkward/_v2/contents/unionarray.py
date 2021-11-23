# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy
import ctypes

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.unionform import UnionForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class UnionArray(Content):
    is_UnionType = True

    def __init__(self, tags, index, contents, identifier=None, parameters=None):
        if not (isinstance(tags, Index) and tags.dtype == np.dtype(np.int8)):
            raise TypeError(
                "{0} 'tags' must be an Index with dtype=int8, not {1}".format(
                    type(self).__name__, repr(tags)
                )
            )
        if not isinstance(index, Index) and index.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                "{0} 'index' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(index))
            )

        if not isinstance(contents, Iterable):
            raise TypeError(
                "{0} 'contents' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)

        for content in contents:
            if not isinstance(content, Content):
                raise TypeError(
                    "{0} all 'contents' must be Content subclasses, not {1}".format(
                        type(self).__name__, repr(content)
                    )
                )

        if not len(tags) <= len(index):
            raise ValueError(
                "{0} len(tags) ({1}) must be <= len(index) ({2})".format(
                    type(self).__name__, len(tags), len(index)
                )
            )

        self._tags = tags
        self._index = index
        self._contents = contents
        self._init(identifier, parameters)

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    def content(self, index):
        return self._contents[index]

    @property
    def contents(self):
        return self._contents

    @property
    def nplike(self):
        return self._tags.nplike

    Form = UnionForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._tags.form,
            self._index.form,
            [x._form_with_key(getkey) for x in self._contents],
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key1 = getkey(self, form, "tags")
        key2 = getkey(self, form, "index")
        container[key1] = ak._v2._util.little_endian(self._tags.to(nplike))
        container[key2] = ak._v2._util.little_endian(self._index.to(nplike))
        for i, content in enumerate(self._contents):
            content._to_buffers(form.content(i), getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return UnionArray(
            ak._v2.index.Index(self._tags.to(tt)),
            ak._v2.index.Index(self._index.to(tt)),
            [x.typetracer for x in self._contents],
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._tags)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnionArray len="]
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._tags._repr(indent + "    ", "<tags>", "</tags>\n"))
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))

        for i, x in enumerate(self._contents):
            out.append("{0}    <content index={1}>\n".format(indent, repr(str(i))))
            out.append(x._repr(indent + "        ", "", "\n"))
            out.append("{0}    </content>\n".format(indent))

        out.append(indent + "</UnionArray>")
        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return UnionArray(
            self._tags,
            self._index,
            self._contents,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
        )

    def _getitem_nothing(self):
        return self._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        tag, index = self._tags[where], self._index[where]
        return self._contents[tag]._getitem_at(index)

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return UnionArray(
            self._tags[start:stop],
            self._index[start:stop],
            self._contents,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return UnionArray(
            self._tags,
            self._index,
            [x._getitem_field(where, only_fields) for x in self._contents],
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return UnionArray(
            self._tags,
            self._index,
            [x._getitem_fields(where, only_fields) for x in self._contents],
            self._fields_identifer(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nexttags = self._tags[carry.data]
            nextindex = self._index[: len(self._tags)][carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return UnionArray(
            nexttags,
            nextindex,
            self._contents,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def project(self, index):
        nplike = self.nplike
        lentags = len(self._tags)
        assert len(self._index) >= lentags
        lenout = ak._v2.index.Index64.empty(1, nplike)
        tmpcarry = ak._v2.index.Index64.empty(lentags, nplike)
        self._handle_error(
            nplike[
                "awkward_UnionArray_project",
                lenout.dtype.type,
                tmpcarry.dtype.type,
                self._tags.dtype.type,
                self._index.dtype.type,
            ](
                lenout.to(nplike),
                tmpcarry.to(nplike),
                self._tags.to(nplike),
                self._index.to(nplike),
                lentags,
                index,
            )
        )
        nextcarry = ak._v2.index.Index64(tmpcarry.data[: lenout[0]], nplike)
        return self._contents[index]._carry(nextcarry, False, NestedIndexError)

    def _regular_index(self, tags):
        nplike = self.nplike
        lentags = len(tags)
        size = ak._v2.index.Index64.empty(1, nplike)
        self._handle_error(
            nplike[
                "awkward_UnionArray_regular_index_getsize",
                size.dtype.type,
                tags.dtype.type,
            ](
                size.to(nplike),
                tags.to(nplike),
                lentags,
            )
        )
        current = ak._v2.index.Index64.empty(size[0], nplike)
        outindex = ak._v2.index.Index64.empty(lentags, nplike)
        self._handle_error(
            nplike[
                "awkward_UnionArray_regular_index",
                outindex.dtype.type,
                current.dtype.type,
                tags.dtype.type,
            ](
                outindex.to(nplike),
                current.to(nplike),
                size[0],
                tags.to(nplike),
                lentags,
            )
        )
        return outindex

    def _nested_tags_index(self, offsets, counts):
        nplike = self.nplike
        f_offsets = ak._v2.index.Index64(copy.deepcopy(offsets.data))
        contentlen = f_offsets[len(f_offsets) - 1]

        tags = ak._v2.index.Index8.empty(contentlen, nplike)
        index = ak._v2.index.Index64.empty(contentlen, nplike)

        for tag in range(len(counts)):
            self._handle_error(
                nplike[
                    "awkward_UnionArray_nestedfill_tags_index",
                    tags.dtype.type,
                    index.dtype.type,
                    f_offsets.dtype.type,
                    counts[tag].dtype.type,
                ](
                    tags.to(nplike),
                    index.to(nplike),
                    f_offsets.to(nplike),
                    tag,
                    counts[tag].to(nplike),
                    len(f_offsets) - 1,
                )
            )
        return (tags, index)

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        simplified = self.simplify_uniontype()
        if (
            simplified.index.dtype == np.dtype(np.int32)
            or simplified.index.dtype == np.dtype(np.uint32)
            or simplified.index.dtype == np.dtype(np.int64)
        ):
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "cannot apply jagged slices to irreducible union arrays",
            )
        return simplified._getitem_next_jagged(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(
            head, (int, slice, ak._v2.index.Index64, ak._v2.contents.ListOffsetArray)
        ):
            outcontents = []
            for i in range(len(self._contents)):
                projection = self.project(i)
                outcontents.append(projection._getitem_next(head, tail, advanced))
            outindex = self._regular_index(self._tags)

            out = UnionArray(
                self._tags,
                outindex,
                outcontents,
                self._identifier,
                self._parameters,
            )
            return out.simplify_uniontype()

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def simplify_uniontype(self, merge=True, mergebool=False):
        tags = ak._v2.index.Index8.empty(len(self), self.nplike)
        index = ak._v2.index.Index64.empty(len(self), self.nplike)
        contents = []
        for i in range(len(self.contents)):
            rawcontent = self.contents[i]
            if isinstance(rawcontent, ak._v2.contents.unionarray.UnionArray):

                innertags = rawcontent.tags
                innerindex = rawcontent.index
                innercontents = rawcontent.contents
                for j in range(len(innercontents)):
                    unmerged = True
                    for k in range(len(contents)):
                        if merge and contents[k].mergeable(innercontents[j], mergebool):
                            self._handle_error(
                                self.nplike[
                                    "awkward_UnionArray_simplify",
                                    tags.dtype.type,
                                    index.dtype.type,
                                    self._tags.dtype.type,
                                    self._index.dtype.type,
                                    innertags.dtype.type,
                                    innerindex.dtype.type,
                                ](
                                    tags.to(self.nplike),
                                    index.to(self.nplike),
                                    self._tags.to(self.nplike),
                                    self._index.to(self.nplike),
                                    innertags.to(self.nplike),
                                    innerindex.to(self.nplike),
                                    k,
                                    j,
                                    i,
                                    len(self),
                                    len(contents[k]),
                                )
                            )
                            contents[k] = contents[k].merge(innercontents[j])
                            unmerged = False
                            break

                    if unmerged:
                        self._handle_error(
                            self.nplike[
                                "awkward_UnionArray_simplify",
                                tags.dtype.type,
                                index.dtype.type,
                                self._tags.dtype.type,
                                self._index.dtype.type,
                                innertags.dtype.type,
                                innerindex.dtype.type,
                            ](
                                tags.to(self.nplike),
                                index.to(self.nplike),
                                self._tags.to(self.nplike),
                                self._index.to(self.nplike),
                                innertags.to(self.nplike),
                                innerindex.to(self.nplike),
                                len(contents),
                                j,
                                i,
                                len(self),
                                0,
                            )
                        )
                        contents.append(innercontents[j])
            else:
                unmerged = True
                for k in range(len(contents)):
                    if contents[k] == self.contents[i]:
                        self._handle_error(
                            self.nplike[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self._tags.dtype.type,
                                self._index.dtype.type,
                            ](
                                tags.to(self.nplike),
                                index.to(self.nplike),
                                self._tags.to(self.nplike),
                                self._index.to(self.nplike),
                                k,
                                i,
                                len(self),
                                0,
                            )
                        )
                        unmerged = False
                        break

                    elif merge and contents[k].mergeable(self.contents[i], mergebool):
                        self._handle_error(
                            self.nplike[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self._tags.dtype.type,
                                self._index.dtype.type,
                            ](
                                tags.to(self.nplike),
                                index.to(self.nplike),
                                self._tags.to(self.nplike),
                                self._index.to(self.nplike),
                                k,
                                i,
                                len(self),
                                len(contents[k]),
                            )
                        )
                        contents[k] = contents[k].merge(self.contents[i])
                        unmerged = False
                        break

                if unmerged:
                    self._handle_error(
                        self.nplike[
                            "awkward_UnionArray_simplify_one",
                            tags.dtype.type,
                            index.dtype.type,
                            self._tags.dtype.type,
                            self._index.dtype.type,
                        ](
                            tags.to(self.nplike),
                            index.to(self.nplike),
                            self._tags.to(self.nplike),
                            self._index.to(self.nplike),
                            len(contents),
                            i,
                            len(self),
                            0,
                        )
                    )

                    contents.append(self.contents[i])

        if len(contents) > 2 ** 7:
            raise AssertionError("FIXME: handle UnionArray with more than 127 contents")

        if len(contents) == 1:
            return contents[0]._carry(index, True, NestedIndexError)

        else:
            return UnionArray(tags, index, contents, self._identifier, self._parameters)

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self.nplike)
            out[0] = len(self)
            return ak._v2.contents.numpyarray.NumpyArray(out)[0]
        else:
            contents = []
            for content in self._contents:
                contents.append(content.num(posaxis, depth))
            out = UnionArray(self._tags, self._index, contents, None, self._parameters)
            return out.simplify_uniontype(True, False)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)

        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")

        else:
            has_offsets = False
            offsetsraws = self.nplike.empty(len(self._contents), dtype=np.intp)
            contents = []

            for i in range(len(self._contents)):
                offsets, flattened = self._contents[i]._offsets_and_flattened(
                    posaxis, depth
                )
                offsetsraws[i] = offsets.ptr
                contents.append(flattened)
                has_offsets = len(offsets) != 0

            if has_offsets:
                total_length = ak._v2.index.Index64.empty(1, self.nplike)
                self._handle_error(
                    self.nplike[
                        "awkward_UnionArray_flatten_length",
                        total_length.dtype.type,
                        self._tags.dtype.type,
                        self._index.dtype.type,
                        np.int64,
                    ](
                        total_length.to(self.nplike),
                        self._tags.to(self.nplike),
                        self._index.to(self.nplike),
                        len(self._tags),
                        offsetsraws.ctypes.data_as(
                            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))
                        ),
                    )
                )

                totags = ak._v2.index.Index8.empty(total_length[0], self.nplike)
                toindex = ak._v2.index.Index64.empty(total_length[0], self.nplike)
                tooffsets = ak._v2.index.Index64.empty(len(self._tags) + 1, self.nplike)

                self._handle_error(
                    self.nplike[
                        "awkward_UnionArray_flatten_combine",
                        totags.dtype.type,
                        toindex.dtype.type,
                        tooffsets.dtype.type,
                        self._tags.dtype.type,
                        self._index.dtype.type,
                        np.int64,
                    ](
                        totags.to(self.nplike),
                        toindex.to(self.nplike),
                        tooffsets.to(self.nplike),
                        self._tags.to(self.nplike),
                        self._index.to(self.nplike),
                        len(self._tags),
                        offsetsraws.ctypes.data_as(
                            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))
                        ),
                    )
                )

                return (
                    tooffsets,
                    UnionArray(totags, toindex, contents, None, self._parameters),
                )

            else:
                offsets = ak._v2.index.Index64.zeros(1, self.nplike, dtype=np.int64)
                return (
                    offsets,
                    UnionArray(
                        self._tags, self._index, contents, None, self._parameters
                    ),
                )

    def mergeable(self, other, mergebool):
        if not _parameters_equal(self._parameters, other._parameters):
            return False

        return True

    def merging_strategy(self, others):
        if len(others) == 0:
            raise ValueError(
                "to merge this array with 'others', at least one other must be provided"
            )

        head = [self]
        tail = []

        for i in range(len(others)):
            head.append(others[i])

        return (head, tail)

    def _reverse_merge(self, other):
        theirlength = len(other)
        mylength = len(self)

        tags = ak._v2.index.Index8.empty(theirlength + mylength, self.nplike)
        index = ak._v2.index.Index64.empty(theirlength + mylength, self.nplike)

        contents = [other]
        contents.extend(self.contents)

        self._handle_error(
            self.nplike["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.to(self.nplike),
                0,
                theirlength,
                0,
            )
        )

        self._handle_error(
            self.nplike["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.to(self.nplike),
                0,
                theirlength,
            )
        )

        self._handle_error(
            self.nplike[
                "awkward_UnionArray_filltags",
                tags.dtype.type,
                self.tags.dtype.type,
            ](
                tags.to(self.nplike),
                theirlength,
                self.tags.to(self.nplike),
                mylength,
                1,
            )
        )

        self._handle_error(
            self.nplike[
                "awkward_UnionArray_fillindex",
                index.dtype.type,
                self.index.dtype.type,
            ](
                index.to(self.nplike),
                theirlength,
                self.index.to(self.nplike),
                mylength,
            )
        )

        if len(contents) > 2 ** 7:
            raise AssertionError("FIXME: handle UnionArray with more than 127 contents")

        parameters = ak._v2._util.merge_parameters(self._parameters, other._parameters)
        return ak._v2.contents.unionarray.UnionArray(
            tags, index, contents, None, parameters
        )

    def mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += len(array)

        nexttags = ak._v2.index.Index8.empty(total_length, self.nplike)
        nextindex = ak._v2.index.Index64.empty(total_length, self.nplike)

        nextcontents = []
        length_so_far = 0
        parameters = {}

        for array in head:
            parameters = ak._v2._util.merge_parameters(
                self._parameters, array._parameters
            )
            if isinstance(array, ak._v2.contents.unionarray.UnionArray):
                union_tags = ak._v2.index.Index(array.tags)
                union_index = ak._v2.index.Index(array.index)
                union_contents = array.contents
                self._handle_error(
                    self.nplike[
                        "awkward_UnionArray_filltags",
                        nexttags.dtype.type,
                        union_tags.dtype.type,
                    ](
                        nexttags.to(self.nplike),
                        length_so_far,
                        union_tags.to(self.nplike),
                        len(array),
                        len(nextcontents),
                    )
                )
                self._handle_error(
                    self.nplike[
                        "awkward_UnionArray_fillindex",
                        nextindex.dtype.type,
                        union_index.dtype.type,
                    ](
                        nextindex.to(self.nplike),
                        length_so_far,
                        union_index.to(self.nplike),
                        len(array),
                    )
                )
                length_so_far += len(array)
                nextcontents.extend(union_contents)

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass

            else:
                self._handle_error(
                    self.nplike[
                        "awkward_UnionArray_filltags_const",
                        nexttags.dtype.type,
                    ](
                        nexttags.to(self.nplike),
                        length_so_far,
                        len(array),
                        len(nextcontents),
                    )
                )

                self._handle_error(
                    self.nplike[
                        "awkward_UnionArray_fillindex_count", nextindex.dtype.type
                    ](nextindex.to(self.nplike), length_so_far, len(array))
                )

                length_so_far += len(array)
                nextcontents.append(array)

        if len(nextcontents) > 127:
            raise ValueError("FIXME: handle UnionArray with more than 127 contents")

        next = ak._v2.contents.unionarray.UnionArray(
            nexttags, nextindex, nextcontents, None, parameters
        )

        # Given UnionArray's merging_strategy, tail is always empty, but just to be formal...

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

    def fillna(self, value):
        contents = []
        for content in self._contents:
            contents.append(content.fillna(value))
        out = UnionArray(
            self._tags, self._index, contents, self._identifier, self._parameters
        )
        return out.simplify_uniontype(True, False)

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._localindex(posaxis, depth))
            return UnionArray(
                self._tags, self._index, contents, self._identifier, self._parameters
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            contents = []
            for content in self._contents:
                contents.append(
                    content._combinations(
                        n, replacement, recordlookup, parameters, posaxis, depth
                    )
                )
            return ak._v2.unionarray.UnionArray(
                self._tags, self._index, contents, self._identifier, self._parameters
            )

    def numbers_to_type(self, name):
        contents = []
        for x in self._contents:
            contents.append(x.numbers_to_type(name))
        return ak._v2.contents.unionarray.UnionArray(
            self._tags, self._index, contents, self._identifier, self._parameters
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        simplified = self.simplify_uniontype(True, True)
        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ValueError("cannot check if an irreducible UnionArray is unique")

        return simplified._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        simplified = self.simplify_uniontype(True, True)
        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ValueError("cannot make a unique irreducible UnionArray")

        return simplified._unique(negaxis, starts, parents, outlength)

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
        if len(self) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        simplified = self.simplify_uniontype(mergebool=True)
        if len(simplified) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ValueError("cannot argsort an irreducible UnionArray")

        return simplified._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable, kind, order
        )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if len(self) == 0:
            return self

        simplified = self.simplify_uniontype(mergebool=True)
        if len(simplified) == 0:
            return simplified

        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ValueError("cannot sort an irreducible UnionArray")

        return simplified._sort_next(
            negaxis, starts, parents, outlength, ascending, stable, kind, order
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
        simplified = self.simplify_uniontype(mergebool=True)
        if isinstance(simplified, UnionArray):
            raise ValueError(
                "cannot call ak.{0} on an irreducible UnionArray".format(reducer.name)
            )

        return simplified._reduce_next(
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
        for i in range(len(self.contents)):
            if isinstance(self.contents[i], ak._v2.contents.unionarray.UnionArray):
                return "{0} contains {1}, the operation that made it might have forgotten to call 'simplify_uniontype'".format(
                    type(self), type(self.contents[i])
                )
            if len(self.index) < len(self.tags):
                return 'at {0} ("{1}"): len(index) < len(tags)'.format(path, type(self))
            lencontents = self.nplike.empty(len(self.contents), dtype=np.int64)
            for i in range(len(self.contents)):
                lencontents[i] = len(self.contents[i])
            error = self.nplike[
                "awkward_UnionArray_validity",
                self.tags.dtype.type,
                self.index.dtype.type,
                np.int64,
            ](
                self.tags.to(self.nplike),
                self.index.to(self.nplike),
                len(self.tags),
                len(self.contents),
                lencontents,
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
            for i in range(len(self.contents)):
                sub = self.contents[i].validityerror(path + ".content({0})".format(i))
                if sub != "":
                    return sub
            return ""

    def _nbytes_part(self):
        result = self.tags._nbytes_part() + self.index._nbytes_part()
        for content in self.contents:
            result = result + content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._rpad(target, posaxis, depth, clip))
            out = ak._v2.contents.unionarray.UnionArray(
                self.tags,
                self.index,
                contents,
                identifier=self._identifier,
                parameters=self._parameters,
            )
            return out.simplify_uniontype(True, False)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        nptags = self._tags.to(numpy)
        npindex = self._index.to(numpy)
        copied_index = False

        values = []
        for tag, content in enumerate(self._contents):
            selected_tags = nptags == tag
            this_index = npindex[selected_tags]

            # Arrow unions can't have masks; propagate validbytes down to the content.
            if validbytes is not None:
                # If this_index is a filtered permutation, we can just filter-permute
                # the mask to have the same order the content.
                if len(numpy.unique(this_index)) == len(this_index):
                    this_validbytes = numpy.zeros(len(this_index), dtype=np.int8)
                    this_validbytes[this_index] = validbytes[selected_tags]

                # If this_index is not a filtered permutation, then we can't modify
                # the mask to fit the content. The same element in the content array
                # will appear multiple times in the union array, and it *might* be
                # presented as valid in some union array elements and masked in others.
                # The validbytes that recurses down to the next level can't have an
                # element that is both 0 (masked) and 1 (valid).
                else:
                    this_validbytes = validbytes[selected_tags]
                    content = content[this_index]
                    if not copied_index:
                        copied_index = True
                        npindex = numpy.array(npindex, copy=True)
                    npindex[selected_tags] = numpy.arange(
                        len(this_index), dtype=npindex.dtype
                    )

            else:
                this_validbytes = None

            values.append(
                content._to_arrow(
                    pyarrow, mask_node, this_validbytes, len(this_index), options
                )
            )

        types = pyarrow.union(
            [
                pyarrow.field(str(i), values[i].type).with_nullable(
                    mask_node is not None or self._contents[i].is_OptionType
                )
                for i in range(len(values))
            ],
            "dense",
            list(range(len(values))),
        )

        if not issubclass(npindex.dtype.type, np.int32):
            npindex = npindex.astype(np.int32)

        return pyarrow.Array.from_buffers(
            ak._v2._connect.pyarrow.to_awkwardarrow_type(
                types, options["extensionarray"], None, self
            ),
            len(nptags),
            [
                None,
                ak._v2._connect.pyarrow.to_length(nptags, length),
                ak._v2._connect.pyarrow.to_length(npindex, length),
            ],
            children=values,
        )

    def _to_numpy(self, allow_missing):
        contents = [
            ak._v2.operations.convert.to_numpy(
                self.project(i), allow_missing=allow_missing
            )
            for i in range(len(self.contents))
        ]

        if any(isinstance(x, self.nplike.ma.MaskedArray) for x in contents):
            try:
                out = self.nplike.ma.concatenate(contents)
            except Exception:
                raise ValueError(
                    "cannot convert {0} into numpy.ma.MaskedArray".format(self)
                )
        else:
            try:
                out = numpy.concatenate(contents)
            except Exception:
                raise ValueError("cannot convert {0} into np.ndarray".format(self))

        tags = numpy.asarray(self.tags)
        for tag, content in enumerate(contents):
            mask = tags == tag
            out[mask] = content
        return out

    def _completely_flatten(self, nplike, options):
        out = []
        for content in self._contents:
            out.extend(content[: len(self._tags)]._completely_flatten(nplike, options))
        return out

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return UnionArray(
                    self._tags,
                    self._index,
                    [
                        content._recursively_apply(
                            action,
                            depth,
                            copy.copy(depth_context),
                            lateral_context,
                            options,
                        )
                        for content in self._contents
                    ],
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                for content in self._contents:
                    content._recursively_apply(
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
        nplike = self._tags.nplike

        tags = self._tags.to(nplike)
        original_index = index = self._index.to(nplike)[: len(tags)]

        contents = list(self._contents)

        for tag in range(len(self._contents)):
            is_tag = tags == tag
            num_tag = nplike.count_nonzero(is_tag)

            if len(contents[tag]) > num_tag:
                if original_index is index:
                    index = index.copy()
                index[is_tag] = nplike.arange(num_tag, dtype=index.dtype)
                contents[tag] = self.project(tag)

            contents[tag] = contents[tag].packed()

        return UnionArray(
            ak._v2.index.Index8(tags),
            ak._v2.index.Index(index),
            contents,
            self._identifier,
            self._parameters,
        )

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        tags = self._tags.to(numpy)
        index = self._index.to(numpy)
        contents = [x._to_list(behavior) for x in self._contents]

        out = [None] * len(tags)
        for i, tag in enumerate(tags):
            out[i] = contents[tag][index[i]]
        return out
