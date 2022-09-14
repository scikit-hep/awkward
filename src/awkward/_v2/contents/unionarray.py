# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# pylint: disable=consider-using-enumerate

import copy
import ctypes

from collections.abc import Iterable

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.index import Index8
from awkward._v2.index import Index64
from awkward._v2.contents.content import Content, unset
from awkward._v2.forms.unionform import UnionForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class UnionArray(Content):
    is_UnionType = True

    def copy(
        self,
        tags=unset,
        index=unset,
        contents=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return UnionArray(
            self._tags if tags is unset else tags,
            self._index if index is unset else index,
            self._contents if contents is unset else contents,
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            tags=copy.deepcopy(self._tags, memo),
            index=copy.deepcopy(self._index, memo),
            contents=[copy.deepcopy(x, memo) for x in self._contents],
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(
        self, tags, index, contents, identifier=None, parameters=None, nplike=None
    ):
        if not (isinstance(tags, Index) and tags.dtype == np.dtype(np.int8)):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'tags' must be an Index with dtype=int8, not {}".format(
                        type(self).__name__, repr(tags)
                    )
                )
            )
        if not isinstance(index, Index) and index.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'index' must be an Index with dtype in (int32, uint32, int64), "
                    "not {}".format(type(self).__name__, repr(index))
                )
            )

        if not isinstance(contents, Iterable):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)

        for content in contents:
            if not isinstance(content, Content):
                raise ak._v2._util.error(
                    TypeError(
                        "{} all 'contents' must be Content subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )

        if (
            tags.nplike.known_shape
            and index.nplike.known_shape
            and tags.length > index.length
        ):
            raise ak._v2._util.error(
                ValueError(
                    "{} len(tags) ({}) must be <= len(index) ({})".format(
                        type(self).__name__, tags.length, index.length
                    )
                )
            )
        if nplike is None:
            for content in contents:
                if nplike is None:
                    nplike = content.nplike
                    break
                elif nplike is not content.nplike:
                    raise ak._v2._util.error(
                        TypeError(
                            "{} 'contents' must use the same array library (nplike): {} vs {}".format(
                                type(self).__name__,
                                type(nplike).__name__,
                                type(content.nplike).__name__,
                            )
                        )
                    )
        if nplike is None:
            nplike = tags.nplike

        self._tags = tags
        self._index = index
        self._contents = contents
        self._init(identifier, parameters, nplike)

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
        container[key1] = ak._v2._util.little_endian(self._tags.raw(nplike))
        container[key2] = ak._v2._util.little_endian(self._index.raw(nplike))
        for i, content in enumerate(self._contents):
            content._to_buffers(form.content(i), getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return UnionArray(
            ak._v2.index.Index(self._tags.raw(tt)),
            ak._v2.index.Index(self._index.raw(tt)),
            [x.typetracer for x in self._contents],
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._tags.length

    def _forget_length(self):
        return UnionArray(
            self._tags.forget_length(),
            self._index,
            self._contents,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnionArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._tags._repr(indent + "    ", "<tags>", "</tags>\n"))
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))

        for i, x in enumerate(self._contents):
            out.append(f"{indent}    <content index={repr(str(i))}>\n")
            out.append(x._repr(indent + "        ", "", "\n"))
            out.append(f"{indent}    </content>\n")

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
            self._nplike,
        )

    def _getitem_nothing(self):
        return self._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return ak._v2._typetracer.OneOf(
                [x._getitem_at(where) for x in self._contents]
            )

        if where < 0:
            where += self.length
        if self._nplike.known_shape and not 0 <= where < self.length:
            raise ak._v2._util.indexerror(self, where)
        tag, index = self._tags[where], self._index[where]
        return self._contents[tag]._getitem_at(index)

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return UnionArray(
            self._tags[start:stop],
            self._index[start:stop],
            self._contents,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return UnionArray(
            self._tags,
            self._index,
            [x._getitem_field(where, only_fields) for x in self._contents],
            self._field_identifier(where),
            None,
            self._nplike,
        ).simplify_uniontype()

    def _getitem_fields(self, where, only_fields=()):
        return UnionArray(
            self._tags,
            self._index,
            [x._getitem_fields(where, only_fields) for x in self._contents],
            self._fields_identifier(where),
            None,
            self._nplike,
        ).simplify_uniontype()

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nexttags = self._tags[carry.data]
            nextindex = self._index[: self._tags.length][carry.data]
        except IndexError as err:
            raise ak._v2._util.indexerror(self, carry.data, str(err)) from err

        return UnionArray(
            nexttags,
            nextindex,
            self._contents,
            self._carry_identifier(carry),
            self._parameters,
            self._nplike,
        )

    def project(self, index):
        lentags = self._tags.length
        assert not self._index.length < lentags
        lenout = ak._v2.index.Index64.empty(1, self._nplike)
        tmpcarry = ak._v2.index.Index64.empty(lentags, self._nplike)
        assert (
            lenout.nplike is self._nplike
            and tmpcarry.nplike is self._nplike
            and self._tags.nplike is self._nplike
            and self._index.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
                "awkward_UnionArray_project",
                lenout.dtype.type,
                tmpcarry.dtype.type,
                self._tags.dtype.type,
                self._index.dtype.type,
            ](
                lenout.data,
                tmpcarry.data,
                self._tags.data,
                self._index.data,
                lentags,
                index,
            )
        )
        nextcarry = ak._v2.index.Index64(
            tmpcarry.data[: lenout[0]], nplike=self._nplike
        )
        return self._contents[index]._carry(nextcarry, False)

    @staticmethod
    def regular_index(tags, IndexClass=Index64, nplike=None):
        if nplike is None:
            nplike = ak.nplike.of(tags)

        lentags = tags.length
        size = ak._v2.index.Index64.empty(1, nplike)
        assert size.nplike is nplike and tags.nplike is nplike
        Content._selfless_handle_error(
            nplike[
                "awkward_UnionArray_regular_index_getsize",
                size.dtype.type,
                tags.dtype.type,
            ](
                size.data,
                tags.data,
                lentags,
            )
        )
        current = IndexClass.empty(size[0], nplike)
        outindex = IndexClass.empty(lentags, nplike)
        assert (
            outindex.nplike is nplike
            and current.nplike is nplike
            and tags.nplike is nplike
        )
        Content._selfless_handle_error(
            nplike[
                "awkward_UnionArray_regular_index",
                outindex.dtype.type,
                current.dtype.type,
                tags.dtype.type,
            ](
                outindex.data,
                current.data,
                size[0],
                tags.data,
                lentags,
            )
        )
        return outindex

    def _regular_index(self, tags):
        return self.regular_index(
            tags, IndexClass=type(self._index), nplike=self._nplike
        )

    @staticmethod
    def nested_tags_index(
        offsets, counts, TagsClass=Index8, IndexClass=Index64, nplike=None
    ):
        if nplike is None:
            nplike = ak.nplike.of(offsets, counts)

        f_offsets = ak._v2.index.Index64(copy.deepcopy(offsets.data))
        contentlen = f_offsets[f_offsets.length - 1]

        tags = TagsClass.empty(contentlen, nplike)
        index = IndexClass.empty(contentlen, nplike)

        for tag, count in enumerate(counts):
            assert (
                tags.nplike is nplike
                and index.nplike is nplike
                and f_offsets.nplike is nplike
                and count.nplike is nplike
            )
            Content._selfless_handle_error(
                nplike[
                    "awkward_UnionArray_nestedfill_tags_index",
                    tags.dtype.type,
                    index.dtype.type,
                    f_offsets.dtype.type,
                    count.dtype.type,
                ](
                    tags.data,
                    index.data,
                    f_offsets.data,
                    tag,
                    count.data,
                    f_offsets.length - 1,
                )
            )
        return (tags, index)

    def _nested_tags_index(self, offsets, counts):
        return self.nested_tags_index(
            offsets,
            counts,
            TagsClass=type(self._tags),
            IndexClass=type(self._index),
            nplike=self._nplike,
        )

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        simplified = self.simplify_uniontype()
        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ak._v2._util.indexerror(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
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
                self._nplike,
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
            raise ak._v2._util.error(AssertionError(repr(head)))

    def simplify_uniontype(self, merge=True, mergebool=False):
        if self._nplike.known_shape and self._index.length < self._tags.length:
            raise ak._v2._util.error(
                ValueError("invalid UnionArray: len(index) < len(tags)")
            )

        length = self._tags.length
        tags = ak._v2.index.Index8.empty(length, self._nplike)
        index = ak._v2.index.Index64.empty(length, self._nplike)
        contents = []

        for i, self_cont in enumerate(self._contents):
            if isinstance(self_cont, UnionArray):
                innertags = self_cont._tags
                innerindex = self_cont._index
                innercontents = self_cont._contents

                for j, inner_cont in enumerate(innercontents):
                    unmerged = True
                    for k in range(len(contents)):
                        if merge and contents[k].mergeable(inner_cont, mergebool):
                            assert (
                                tags.nplike is self._nplike
                                and index.nplike is self._nplike
                                and self._tags.nplike is self._nplike
                                and self._index.nplike is self._nplike
                                and innertags.nplike is self._nplike
                                and innerindex.nplike is self._nplike
                            )
                            self._handle_error(
                                self._nplike[
                                    "awkward_UnionArray_simplify",
                                    tags.dtype.type,
                                    index.dtype.type,
                                    self._tags.dtype.type,
                                    self._index.dtype.type,
                                    innertags.dtype.type,
                                    innerindex.dtype.type,
                                ](
                                    tags.data,
                                    index.data,
                                    self._tags.data,
                                    self._index.data,
                                    innertags.data,
                                    innerindex.data,
                                    k,
                                    j,
                                    i,
                                    length,
                                    contents[k].length,
                                )
                            )
                            contents[k] = contents[k].merge(inner_cont)
                            unmerged = False
                            break

                    if unmerged:
                        assert (
                            tags.nplike is self._nplike
                            and index.nplike is self._nplike
                            and self._tags.nplike is self._nplike
                            and self._index.nplike is self._nplike
                            and innertags.nplike is self._nplike
                            and innerindex.nplike is self._nplike
                        )
                        self._handle_error(
                            self._nplike[
                                "awkward_UnionArray_simplify",
                                tags.dtype.type,
                                index.dtype.type,
                                self._tags.dtype.type,
                                self._index.dtype.type,
                                innertags.dtype.type,
                                innerindex.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self._tags.data,
                                self._index.data,
                                innertags.data,
                                innerindex.data,
                                len(contents),
                                j,
                                i,
                                length,
                                0,
                            )
                        )
                        contents.append(inner_cont)

            else:
                unmerged = True
                for k in range(len(contents)):
                    if contents[k] is self_cont:
                        assert (
                            tags.nplike is self._nplike
                            and index.nplike is self._nplike
                            and self._tags.nplike is self._nplike
                            and self._index.nplike is self._nplike
                        )
                        self._handle_error(
                            self._nplike[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self._tags.dtype.type,
                                self._index.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self._tags.data,
                                self._index.data,
                                k,
                                i,
                                length,
                                0,
                            )
                        )
                        unmerged = False
                        break

                    elif merge and contents[k].mergeable(self_cont, mergebool):
                        assert (
                            tags.nplike is self._nplike
                            and index.nplike is self._nplike
                            and self._tags.nplike is self._nplike
                            and self._index.nplike is self._nplike
                        )
                        self._handle_error(
                            self._nplike[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self._tags.dtype.type,
                                self._index.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self._tags.data,
                                self._index.data,
                                k,
                                i,
                                length,
                                contents[k].length,
                            )
                        )
                        contents[k] = contents[k].merge(self_cont)
                        unmerged = False
                        break

                if unmerged:
                    assert (
                        tags.nplike is self._nplike
                        and index.nplike is self._nplike
                        and self._tags.nplike is self._nplike
                        and self._index.nplike is self._nplike
                    )
                    self._handle_error(
                        self._nplike[
                            "awkward_UnionArray_simplify_one",
                            tags.dtype.type,
                            index.dtype.type,
                            self._tags.dtype.type,
                            self._index.dtype.type,
                        ](
                            tags.data,
                            index.data,
                            self._tags.data,
                            self._index.data,
                            len(contents),
                            i,
                            length,
                            0,
                        )
                    )
                    contents.append(self_cont)

        if len(contents) > 2**7:
            raise ak._v2._util.error(
                NotImplementedError(
                    "FIXME: handle UnionArray with more than 127 contents"
                )
            )

        if len(contents) == 1:
            return contents[0]._carry(index, True)

        else:
            return UnionArray(
                tags, index, contents, self._identifier, self._parameters, self._nplike
            )

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = self.length
            if ak._v2._util.isint(out):
                return np.int64(out)
            else:
                return out
        else:
            contents = []
            for content in self._contents:
                contents.append(content.num(posaxis, depth))
            out = UnionArray(
                self._tags, self._index, contents, None, self._parameters, self._nplike
            )
            return out.simplify_uniontype(True, False)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)

        if posaxis == depth:
            raise ak._v2._util.error(np.AxisError("axis=0 not allowed for flatten"))

        else:
            has_offsets = False
            offsetsraws = self._nplike.index_nplike.empty(
                len(self._contents), dtype=np.intp
            )
            contents = []

            for i in range(len(self._contents)):
                offsets, flattened = self._contents[i]._offsets_and_flattened(
                    posaxis, depth
                )
                offsetsraws[i] = offsets.ptr
                contents.append(flattened)
                has_offsets = offsets.length != 0

            if has_offsets:
                total_length = ak._v2.index.Index64.empty(1, self._nplike)
                assert (
                    total_length.nplike is self._nplike
                    and self._tags.nplike is self._nplike
                    and self._index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_UnionArray_flatten_length",
                        total_length.dtype.type,
                        self._tags.dtype.type,
                        self._index.dtype.type,
                        np.int64,
                    ](
                        total_length.data,
                        self._tags.data,
                        self._index.data,
                        self._tags.length,
                        offsetsraws.ctypes.data_as(
                            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))
                        ),
                    )
                )

                totags = ak._v2.index.Index8.empty(total_length[0], self._nplike)
                toindex = ak._v2.index.Index64.empty(total_length[0], self._nplike)
                tooffsets = ak._v2.index.Index64.empty(
                    self._tags.length + 1, self._nplike
                )

                assert (
                    totags.nplike is self._nplike
                    and toindex.nplike is self._nplike
                    and tooffsets.nplike is self._nplike
                    and self._tags.nplike is self._nplike
                    and self._index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_UnionArray_flatten_combine",
                        totags.dtype.type,
                        toindex.dtype.type,
                        tooffsets.dtype.type,
                        self._tags.dtype.type,
                        self._index.dtype.type,
                        np.int64,
                    ](
                        totags.data,
                        toindex.data,
                        tooffsets.data,
                        self._tags.data,
                        self._index.data,
                        self._tags.length,
                        offsetsraws.ctypes.data_as(
                            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))
                        ),
                    )
                )

                return (
                    tooffsets,
                    UnionArray(
                        totags, toindex, contents, None, self._parameters, self._nplike
                    ),
                )

            else:
                offsets = ak._v2.index.Index64.zeros(0, self._nplike, dtype=np.int64)
                return (
                    offsets,
                    UnionArray(
                        self._tags,
                        self._index,
                        contents,
                        None,
                        self._parameters,
                        self._nplike,
                    ),
                )

    def _mergeable(self, other, mergebool):
        return True

    def merging_strategy(self, others):
        if len(others) == 0:
            raise ak._v2._util.error(
                ValueError(
                    "to merge this array with 'others', at least one other must be provided"
                )
            )

        head = [self]
        tail = []

        for i in range(len(others)):
            head.append(others[i])

        if any(
            isinstance(x.nplike, ak._v2._typetracer.TypeTracer) for x in head + tail
        ):
            head = [
                x
                if isinstance(x.nplike, ak._v2._typetracer.TypeTracer)
                else x.typetracer
                for x in head
            ]
            tail = [
                x
                if isinstance(x.nplike, ak._v2._typetracer.TypeTracer)
                else x.typetracer
                for x in tail
            ]

        return (head, tail)

    def _reverse_merge(self, other):
        theirlength = other.length
        mylength = self.length

        tags = ak._v2.index.Index8.empty(theirlength + mylength, self._nplike)
        index = ak._v2.index.Index64.empty(theirlength + mylength, self._nplike)

        contents = [other]
        contents.extend(self.contents)

        assert tags.nplike is self._nplike
        self._handle_error(
            self._nplike["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data,
                0,
                theirlength,
                0,
            )
        )

        assert index.nplike is self._nplike
        self._handle_error(
            self._nplike["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data,
                0,
                theirlength,
            )
        )

        assert tags.nplike is self._nplike and self.tags.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_UnionArray_filltags",
                tags.dtype.type,
                self.tags.dtype.type,
            ](
                tags.data,
                theirlength,
                self.tags.data,
                mylength,
                1,
            )
        )

        assert index.nplike is self._nplike and self.index.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_UnionArray_fillindex",
                index.dtype.type,
                self.index.dtype.type,
            ](
                index.data,
                theirlength,
                self.index.data,
                mylength,
            )
        )

        if len(contents) > 2**7:
            raise ak._v2._util.error(
                AssertionError("FIXME: handle UnionArray with more than 127 contents")
            )

        parameters = ak._v2._util.merge_parameters(self._parameters, other._parameters)
        return ak._v2.contents.unionarray.UnionArray(
            tags, index, contents, None, parameters, self._nplike
        )

    def mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        nexttags = ak._v2.index.Index8.empty(total_length, self._nplike)
        nextindex = ak._v2.index.Index64.empty(total_length, self._nplike)

        nextcontents = []
        length_so_far = 0
        parameters = self._parameters

        parameters = self._parameters
        for array in head:
            parameters = ak._v2._util.merge_parameters(
                parameters, array._parameters, True
            )
            if isinstance(array, ak._v2.contents.unionarray.UnionArray):
                union_tags = ak._v2.index.Index(array.tags)
                union_index = ak._v2.index.Index(array.index)
                union_contents = array.contents
                assert (
                    nexttags.nplike is self._nplike
                    and union_tags.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_UnionArray_filltags",
                        nexttags.dtype.type,
                        union_tags.dtype.type,
                    ](
                        nexttags.data,
                        length_so_far,
                        union_tags.data,
                        array.length,
                        len(nextcontents),
                    )
                )
                assert (
                    nextindex.nplike is self._nplike
                    and union_index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_UnionArray_fillindex",
                        nextindex.dtype.type,
                        union_index.dtype.type,
                    ](
                        nextindex.data,
                        length_so_far,
                        union_index.data,
                        array.length,
                    )
                )
                length_so_far += array.length
                nextcontents.extend(union_contents)

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass

            else:
                assert nexttags.nplike is self._nplike
                self._handle_error(
                    self._nplike[
                        "awkward_UnionArray_filltags_const",
                        nexttags.dtype.type,
                    ](
                        nexttags.data,
                        length_so_far,
                        array.length,
                        len(nextcontents),
                    )
                )

                assert nextindex.nplike is self._nplike
                self._handle_error(
                    self._nplike[
                        "awkward_UnionArray_fillindex_count", nextindex.dtype.type
                    ](nextindex.data, length_so_far, array.length)
                )

                length_so_far += array.length
                nextcontents.append(array)

        if len(nextcontents) > 127:
            raise ak._v2._util.error(
                ValueError("FIXME: handle UnionArray with more than 127 contents")
            )

        next = ak._v2.contents.unionarray.UnionArray(
            nexttags, nextindex, nextcontents, None, parameters, self._nplike
        )

        # Given UnionArray's merging_strategy, tail is always empty, but just to be formal...

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

    def fill_none(self, value):
        contents = []
        for content in self._contents:
            contents.append(content.fill_none(value))
        out = UnionArray(
            self._tags,
            self._index,
            contents,
            self._identifier,
            self._parameters,
            self._nplike,
        )
        return out.simplify_uniontype(True, False)

    def _local_index(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._local_index_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._local_index(posaxis, depth))
            return UnionArray(
                self._tags,
                self._index,
                contents,
                self._identifier,
                self._parameters,
                self._nplike,
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
                self._tags,
                self._index,
                contents,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def numbers_to_type(self, name):
        contents = []
        for x in self._contents:
            contents.append(x.numbers_to_type(name))
        return ak._v2.contents.unionarray.UnionArray(
            self._tags,
            self._index,
            contents,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        simplified = self.simplify_uniontype(True, True)
        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ak._v2._util.error(
                ValueError("cannot check if an irreducible UnionArray is unique")
            )

        return simplified._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        simplified = self.simplify_uniontype(True, True)
        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ak._v2._util.error(
                ValueError("cannot make a unique irreducible UnionArray")
            )

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
        simplified = self.simplify_uniontype(mergebool=True)
        if simplified.length == 0:
            return ak._v2.contents.NumpyArray(
                self._nplike.empty(0, np.int64), None, None, self._nplike
            )

        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ak._v2._util.error(
                ValueError("cannot argsort an irreducible UnionArray")
            )

        return simplified._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable, kind, order
        )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if self.length == 0:
            return self

        simplified = self.simplify_uniontype(mergebool=True)
        if simplified.length == 0:
            return simplified

        if isinstance(simplified, ak._v2.contents.UnionArray):
            raise ak._v2._util.error(
                ValueError("cannot sort an irreducible UnionArray")
            )

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
        behavior,
    ):
        simplified = self.simplify_uniontype(mergebool=True)
        if isinstance(simplified, UnionArray):
            raise ak._v2._util.error(
                ValueError(
                    f"cannot call ak.{reducer.name} on an irreducible UnionArray"
                )
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
            behavior,
        )

    def _validity_error(self, path):
        for i in range(len(self.contents)):
            if isinstance(self.contents[i], ak._v2.contents.unionarray.UnionArray):
                return "{} contains {}, the operation that made it might have forgotten to call 'simplify_uniontype'".format(
                    type(self), type(self.contents[i])
                )
            if self._nplike.known_shape and self.index.length < self.tags.length:
                return f'at {path} ("{type(self)}"): len(index) < len(tags)'

            lencontents = self._nplike.index_nplike.empty(
                len(self.contents), dtype=np.int64
            )
            if self._nplike.known_shape:
                for i in range(len(self.contents)):
                    lencontents[i] = self.contents[i].length

            error = self._nplike[
                "awkward_UnionArray_validity",
                self.tags.dtype.type,
                self.index.dtype.type,
                np.int64,
            ](
                self.tags.data,
                self.index.data,
                self.tags.length,
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
                return 'at {} ("{}"): {} at i={}{}'.format(
                    path, type(self), message, error.id, filename
                )

            for i in range(len(self.contents)):
                sub = self.contents[i].validity_error(path + f".content({i})")
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

    def _pad_none(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.pad_none_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._pad_none(target, posaxis, depth, clip))
            out = ak._v2.contents.unionarray.UnionArray(
                self.tags,
                self.index,
                contents,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return out.simplify_uniontype(True, False)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        nptags = self._tags.raw(numpy)
        npindex = self._index.raw(numpy)
        copied_index = False

        values = []
        for tag, content in enumerate(self._contents):
            selected_tags = nptags == tag
            this_index = npindex[selected_tags]

            # Arrow unions can't have masks; propagate validbytes down to the content.
            if validbytes is not None:
                # If this_index is a filtered permutation, we can just filter-permute
                # the mask to have the same order the content.
                if numpy.unique(this_index).shape[0] == this_index.shape[0]:
                    this_validbytes = numpy.zeros(this_index.shape[0], dtype=np.int8)
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
                        this_index.shape[0], dtype=npindex.dtype
                    )

            else:
                this_validbytes = None

            values.append(
                content._to_arrow(
                    pyarrow, mask_node, this_validbytes, this_index.shape[0], options
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
                types,
                options["extensionarray"],
                options["record_is_scalar"],
                None,
                self,
            ),
            nptags.shape[0],
            [
                None,
                ak._v2._connect.pyarrow.to_length(nptags, length),
                ak._v2._connect.pyarrow.to_length(npindex, length),
            ],
            children=values,
        )

    def _to_numpy(self, allow_missing):
        contents = [
            ak._v2.operations.to_numpy(self.project(i), allow_missing=allow_missing)
            for i in range(len(self.contents))
        ]

        if any(isinstance(x, self._nplike.ma.MaskedArray) for x in contents):
            try:
                out = self._nplike.ma.concatenate(contents)
            except Exception as err:
                raise ak._v2._util.error(
                    ValueError(f"cannot convert {self} into numpy.ma.MaskedArray")
                ) from err
        else:
            try:
                out = numpy.concatenate(contents)
            except Exception as err:
                raise ak._v2._util.error(
                    ValueError(f"cannot convert {self} into np.ndarray")
                ) from err

        tags = numpy.asarray(self.tags)
        for tag, content in enumerate(contents):
            mask = tags == tag
            out[mask] = content
        return out

    def _completely_flatten(self, nplike, options):
        out = []
        for i in range(len(self._contents)):
            index = self._index[self._tags.data == i]
            out.extend(
                self._contents[i]
                ._carry(index, False)
                ._completely_flatten(nplike, options)
            )
        return out

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return UnionArray(
                    self._tags,
                    self._index,
                    [
                        content._recursively_apply(
                            action,
                            behavior,
                            depth,
                            copy.copy(depth_context),
                            lateral_context,
                            options,
                        )
                        for content in self._contents
                    ],
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                    self._nplike,
                )

        else:

            def continuation():
                for content in self._contents:
                    content._recursively_apply(
                        action,
                        behavior,
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
            behavior=behavior,
            nplike=self._nplike,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._v2._util.error(AssertionError(result))

    def packed(self):
        tags = self._tags.raw(self._nplike)
        original_index = index = self._index.raw(self._nplike)[: tags.shape[0]]

        contents = list(self._contents)

        for tag in range(len(self._contents)):
            is_tag = tags == tag
            num_tag = self._nplike.index_nplike.count_nonzero(is_tag)

            if len(contents[tag]) > num_tag:
                if original_index is index:
                    index = index.copy()
                index[is_tag] = self._nplike.index_nplike.arange(
                    num_tag, dtype=index.dtype
                )
                contents[tag] = self.project(tag)

            contents[tag] = contents[tag].packed()

        return UnionArray(
            ak._v2.index.Index8(tags, nplike=self.nplike),
            ak._v2.index.Index(index, nplike=self.nplike),
            contents,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        tags = self._tags.raw(numpy)
        index = self._index.raw(numpy)
        contents = [x._to_list(behavior, json_conversions) for x in self._contents]

        out = [None] * tags.shape[0]
        for i, tag in enumerate(tags):
            out[i] = contents[tag][index[i]]
        return out

    def _to_nplike(self, nplike):
        index = self._index._to_nplike(nplike)
        contents = [content._to_nplike(nplike) for content in self._contents]
        return UnionArray(
            self._tags,
            index,
            contents,
            identifier=self.identifier,
            parameters=self.parameters,
            nplike=nplike,
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return (
            self.tags == other.tags
            and self.index.layout_equal(other.index, index_dtype, numpyarray)
            and len(self.contents) == len(other.contents)
            and all(
                [
                    self.contents[i].layout_equal(
                        other.contents[i], index_dtype, numpyarray
                    )
                    for i in range(len(self.contents))
                ]
            )
        )
