# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# pylint: disable=consider-using-enumerate
from __future__ import annotations

import copy
import ctypes
from collections.abc import Iterable, Sequence

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.unionform import UnionForm
from awkward.index import Index, Index8, Index64
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class UnionArray(Content):
    is_union = True

    def __init__(self, tags, index, contents, *, parameters=None):
        if not (isinstance(tags, Index) and tags.dtype == np.dtype(np.int8)):
            raise ak._errors.wrap_error(
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
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'index' must be an Index with dtype in (int32, uint32, int64), "
                    "not {}".format(type(self).__name__, repr(index))
                )
            )

        if not isinstance(contents, Iterable):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)

        if len(contents) < 2:
            raise ak._errors.wrap_error(
                TypeError(f"{type(self).__name__} must have at least 2 'contents'")
            )
        for content in contents:
            if not isinstance(content, Content):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} all 'contents' must be Content subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
            if content.is_union:
                raise ak._errors.wrap_error(
                    TypeError(
                        "{0} cannot contain union-types in its 'contents' ({1}); try {0}.simplified instead".format(
                            type(self).__name__, type(content).__name__
                        )
                    )
                )

        for content in contents[1:]:
            if contents[0]._mergeable(content, mergebool=False):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{0} cannot contain mergeable 'contents' ({1} of {2} and {3} of {4}); try {0}.simplified instead".format(
                            type(self).__name__,
                            type(contents[0]).__name__,
                            repr(str(contents[0].form.type)),
                            type(content).__name__,
                            repr(str(content.form.type)),
                        )
                    )
                )

        if (
            tags.nplike.known_shape
            and index.nplike.known_shape
            and tags.length > index.length
        ):
            raise ak._errors.wrap_error(
                ValueError(
                    "{} len(tags) ({}) must be <= len(index) ({})".format(
                        type(self).__name__, tags.length, index.length
                    )
                )
            )

        backend = None
        for content in contents:
            if backend is None:
                backend = content.backend
            elif backend is not content.backend:
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} 'contents' must use the same array library (backend): {} vs {}".format(
                            type(self).__name__,
                            type(backend).__name__,
                            type(content.backend).__name__,
                        )
                    )
                )

        assert tags.nplike is backend.index_nplike
        assert index.nplike is backend.index_nplike

        self._tags = tags
        self._index = index
        self._contents = contents
        self._init(parameters, backend)

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    form_cls: Final = UnionForm

    def copy(
        self,
        tags=unset,
        index=unset,
        contents=unset,
        *,
        parameters=unset,
    ):
        return UnionArray(
            self._tags if tags is unset else tags,
            self._index if index is unset else index,
            self._contents if contents is unset else contents,
            parameters=self._parameters if parameters is unset else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            tags=copy.deepcopy(self._tags, memo),
            index=copy.deepcopy(self._index, memo),
            contents=[copy.deepcopy(x, memo) for x in self._contents],
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(
        cls,
        tags,
        index,
        contents,
        *,
        parameters=None,
        merge=True,
        mergebool=False,
    ):
        self_index = index
        self_tags = tags
        self_contents = contents

        backend = None
        for content in contents:
            if backend is None:
                backend = content.backend
            elif backend is not content.backend:
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} 'contents' must use the same array library (backend): {} vs {}".format(
                            cls.__name__,
                            type(backend).__name__,
                            type(content.backend).__name__,
                        )
                    )
                )

        if backend.nplike.known_shape and self_index.length < self_tags.length:
            raise ak._errors.wrap_error(
                ValueError("invalid UnionArray: len(index) < len(tags)")
            )

        length = self_tags.length
        tags = ak.index.Index8.empty(length, backend.index_nplike)
        index = ak.index.Index64.empty(length, backend.index_nplike)
        contents = []

        for i, self_cont in enumerate(self_contents):
            if isinstance(self_cont, UnionArray):
                innertags = self_cont._tags
                innerindex = self_cont._index
                innercontents = self_cont._contents

                for j, inner_cont in enumerate(innercontents):
                    unmerged = True
                    for k in range(len(contents)):
                        if merge and contents[k]._mergeable(inner_cont, mergebool):
                            Content._selfless_handle_error(
                                backend[
                                    "awkward_UnionArray_simplify",
                                    tags.dtype.type,
                                    index.dtype.type,
                                    self_tags.dtype.type,
                                    self_index.dtype.type,
                                    innertags.dtype.type,
                                    innerindex.dtype.type,
                                ](
                                    tags.data,
                                    index.data,
                                    self_tags.data,
                                    self_index.data,
                                    innertags.data,
                                    innerindex.data,
                                    k,
                                    j,
                                    i,
                                    length,
                                    contents[k].length,
                                )
                            )
                            old_parameters = contents[k]._parameters
                            contents[k] = (
                                contents[k]
                                ._mergemany([inner_cont])
                                .copy(
                                    parameters=ak._util.merge_parameters(
                                        old_parameters, inner_cont._parameters
                                    )
                                )
                            )
                            unmerged = False
                            break

                    if unmerged:
                        Content._selfless_handle_error(
                            backend[
                                "awkward_UnionArray_simplify",
                                tags.dtype.type,
                                index.dtype.type,
                                self_tags.dtype.type,
                                self_index.dtype.type,
                                innertags.dtype.type,
                                innerindex.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self_tags.data,
                                self_index.data,
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
                        Content._selfless_handle_error(
                            backend[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self_tags.dtype.type,
                                self_index.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self_tags.data,
                                self_index.data,
                                k,
                                i,
                                length,
                                0,
                            )
                        )
                        unmerged = False
                        break

                    elif merge and contents[k]._mergeable(self_cont, mergebool):
                        Content._selfless_handle_error(
                            backend[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self_tags.dtype.type,
                                self_index.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self_tags.data,
                                self_index.data,
                                k,
                                i,
                                length,
                                contents[k].length,
                            )
                        )
                        old_parameters = contents[k]._parameters
                        contents[k] = (
                            contents[k]
                            ._mergemany([self_cont])
                            .copy(
                                parameters=ak._util.merge_parameters(
                                    old_parameters, self_cont._parameters
                                )
                            )
                        )
                        unmerged = False
                        break

                if unmerged:
                    Content._selfless_handle_error(
                        backend[
                            "awkward_UnionArray_simplify_one",
                            tags.dtype.type,
                            index.dtype.type,
                            self_tags.dtype.type,
                            self_index.dtype.type,
                        ](
                            tags.data,
                            index.data,
                            self_tags.data,
                            self_index.data,
                            len(contents),
                            i,
                            length,
                            0,
                        )
                    )
                    contents.append(self_cont)

        if len(contents) > 2**7:
            raise ak._errors.wrap_error(
                NotImplementedError(
                    "FIXME: handle UnionArray with more than 127 contents"
                )
            )

        if len(contents) == 1:
            next = contents[0]._carry(index, True)
            return next.copy(
                parameters=ak._util.merge_parameters(next._parameters, parameters)
            )

        else:
            return cls(
                tags,
                index,
                contents,
                parameters=parameters,
            )

    def content(self, index):
        return self._contents[index]

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            self._tags.form,
            self._index.form,
            [x._form_with_key(getkey) for x in self._contents],
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        key1 = getkey(self, form, "tags")
        key2 = getkey(self, form, "index")
        container[key1] = ak._util.little_endian(self._tags.raw(backend.index_nplike))
        container[key2] = ak._util.little_endian(self._index.raw(backend.index_nplike))
        for i, content in enumerate(self._contents):
            content._to_buffers(form.content(i), getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        tt = ak._typetracer.TypeTracer.instance()
        tags = self._tags.to_nplike(tt)
        return UnionArray(
            tags.forget_length() if forget_length else tags,
            self._index.to_nplike(tt),
            [x._to_typetracer(False) for x in self._contents],
            parameters=self._parameters,
        )

    def _touch_data(self, recursive):
        if not self._backend.index_nplike.known_data:
            self._tags.data.touch_data()
            self._index.data.touch_data()
        if recursive:
            for x in self._contents:
                x._touch_data(recursive)

    def _touch_shape(self, recursive):
        if not self._backend.index_nplike.known_shape:
            self._tags.data.touch_shape()
            self._index.data.touch_shape()
        if recursive:
            for x in self._contents:
                x._touch_shape(recursive)

    @property
    def length(self):
        return self._tags.length

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

    def _getitem_nothing(self):
        return self._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return ak._typetracer.OneOf([x._getitem_at(where) for x in self._contents])

        if where < 0:
            where += self.length
        if self._backend.nplike.known_shape and not 0 <= where < self.length:
            raise ak._errors.index_error(self, where)
        tag, index = self._tags[where], self._index[where]
        return self._contents[tag]._getitem_at(index)

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return UnionArray(
            self._tags[start:stop],
            self._index[start:stop],
            self._contents,
            parameters=self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return UnionArray.simplified(
            self._tags,
            self._index,
            [x._getitem_field(where, only_fields) for x in self._contents],
            parameters=None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return UnionArray.simplified(
            self._tags,
            self._index,
            [x._getitem_fields(where, only_fields) for x in self._contents],
            parameters=None,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)

        try:
            nexttags = self._tags[carry.data]
            nextindex = self._index[: self._tags.length][carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return UnionArray(
            nexttags,
            nextindex,
            self._contents,
            parameters=self._parameters,
        )

    def _union_of_optionarrays(self, index, parameters):
        tag_for_missing = 0
        for i, content in enumerate(self._contents):
            if content.is_option:
                tag_for_missing = i
                break

        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            nexttags = self._tags.data
            nextindex = self._index.data
            contents = []
            for tag, content in enumerate(self._contents):
                if tag == tag_for_missing:
                    indexedoption_index = self._backend.index_nplike.arange(
                        content.length + 1, dtype=np.int64
                    )
                    contents.append(
                        ak.contents.IndexedOptionArray.simplified(
                            ak.index.Index64(indexedoption_index), content
                        )
                    )
                else:
                    contents.append(ak.contents.UnmaskedArray.simplified(content))

        else:
            # like _carry, above
            carry_data = index.raw(self._backend.nplike).copy()
            is_missing = carry_data < 0

            if self._tags.length != 0:
                # but the missing values will temporarily use 0 as a placeholder
                carry_data[is_missing] = 0
                try:
                    nexttags = self._tags.data[carry_data]
                    nextindex = self._index.data[: self._tags.length][carry_data]
                except IndexError as err:
                    raise ak._errors.index_error(self, carry_data, str(err)) from err

                # now actually set the missing values
                nexttags[is_missing] = tag_for_missing
                nextindex[is_missing] = self._contents[tag_for_missing].length

            else:
                # UnionArray is empty, so
                nexttags = self._backend.index_nplike.full(
                    len(carry_data), tag_for_missing, dtype=self._tags.dtype
                )
                nextindex = self._backend.index_nplike.full(
                    len(carry_data),
                    self._contents[tag_for_missing].length,
                    dtype=self._index.dtype,
                )

            contents = []
            for tag, content in enumerate(self._contents):
                if tag == tag_for_missing:
                    indexedoption_index = self._backend.index_nplike.arange(
                        content.length + 1, dtype=np.int64
                    )
                    indexedoption_index[content.length] = -1
                    contents.append(
                        ak.contents.IndexedOptionArray.simplified(
                            ak.index.Index64(indexedoption_index), content
                        )
                    )
                else:
                    contents.append(ak.contents.UnmaskedArray.simplified(content))

        return UnionArray.simplified(
            ak.index.Index(nexttags),
            ak.index.Index(nextindex),
            contents,
            parameters=ak._util.merge_parameters(self._parameters, parameters),
        )

    def project(self, index):
        lentags = self._tags.length
        assert not self._index.length < lentags
        lenout = ak.index.Index64.empty(1, self._backend.index_nplike)
        tmpcarry = ak.index.Index64.empty(lentags, self._backend.index_nplike)
        assert (
            lenout.nplike is self._backend.index_nplike
            and tmpcarry.nplike is self._backend.index_nplike
            and self._tags.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
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
        nextcarry = ak.index.Index64(
            tmpcarry.data[: lenout[0]], nplike=self._backend.index_nplike
        )
        return self._contents[index]._carry(nextcarry, False)

    @staticmethod
    def regular_index(
        tags: Index,
        *,
        backend: ak._backends.Backend,
        index_cls: type[Index] = Index64,
    ):
        tags = tags.to_nplike(backend.index_nplike)

        lentags = tags.length
        size = ak.index.Index64.empty(1, nplike=backend.index_nplike)
        assert (
            size.nplike is backend.index_nplike and tags.nplike is backend.index_nplike
        )
        Content._selfless_handle_error(
            backend[
                "awkward_UnionArray_regular_index_getsize",
                size.dtype.type,
                tags.dtype.type,
            ](
                size.data,
                tags.data,
                lentags,
            )
        )
        current = index_cls.empty(size[0], nplike=backend.index_nplike)
        outindex = index_cls.empty(lentags, nplike=backend.index_nplike)
        assert (
            outindex.nplike is backend.index_nplike
            and current.nplike is backend.index_nplike
            and tags.nplike is backend.index_nplike
        )
        Content._selfless_handle_error(
            backend[
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

    def _regular_index(self, tags: Index) -> Index:
        return self.regular_index(
            tags, index_cls=type(self._index), backend=self._backend
        )

    @staticmethod
    def nested_tags_index(
        offsets: Index,
        counts: Sequence[Index],
        *,
        backend: ak._backends.Backend,
        tags_cls: type[Index] = Index8,
        index_cls: type[Index] = Index64,
    ) -> tuple[Index, Index]:
        offsets = offsets.to_nplike(backend.index_nplike)
        counts = [c.to_nplike(backend.index_nplike) for c in counts]

        f_offsets = ak.index.Index64(copy.deepcopy(offsets.data))
        contentlen = f_offsets[f_offsets.length - 1]

        tags = tags_cls.empty(contentlen, nplike=backend.index_nplike)
        index = index_cls.empty(contentlen, nplike=backend.index_nplike)

        for tag, count in enumerate(counts):
            assert (
                tags.nplike is backend.index_nplike
                and index.nplike is backend.index_nplike
                and f_offsets.nplike is backend.index_nplike
                and count.nplike is backend.index_nplike
            )
            Content._selfless_handle_error(
                backend[
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

    def _nested_tags_index(self, offsets: Index, counts: Sequence[Index]):
        return self.nested_tags_index(
            offsets,
            counts,
            backend=self._backend,
            tags_cls=type(self._tags),
            index_cls=type(self._index),
        )

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if isinstance(self, ak.contents.UnionArray):
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                "cannot apply jagged slices to irreducible union arrays",
            )
        return self._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(
            head, (int, slice, ak.index.Index64, ak.contents.ListOffsetArray)
        ):
            outcontents = []
            for i in range(len(self._contents)):
                projection = self.project(i)
                outcontents.append(projection._getitem_next(head, tail, advanced))
            outindex = self._regular_index(self._tags)

            return UnionArray.simplified(
                self._tags,
                outindex,
                outcontents,
                parameters=self._parameters,
            )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)

        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(np.AxisError("axis=0 not allowed for flatten"))

        else:
            has_offsets = False
            offsetsraws = self._backend.index_nplike.empty(
                len(self._contents), dtype=np.intp
            )
            contents = []

            for i in range(len(self._contents)):
                offsets, flattened = self._contents[i]._offsets_and_flattened(
                    axis, depth
                )
                offsetsraws[i] = offsets.ptr
                contents.append(flattened)
                has_offsets = offsets.length != 0

            if has_offsets:
                total_length = ak.index.Index64.empty(
                    1, nplike=self._backend.index_nplike
                )
                assert (
                    total_length.nplike is self._backend.index_nplike
                    and self._tags.nplike is self._backend.index_nplike
                    and self._index.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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

                totags = ak.index.Index8.empty(
                    total_length[0], nplike=self._backend.index_nplike
                )
                toindex = ak.index.Index64.empty(
                    total_length[0], nplike=self._backend.index_nplike
                )
                tooffsets = ak.index.Index64.empty(
                    self._tags.length + 1, nplike=self._backend.index_nplike
                )

                assert (
                    totags.nplike is self._backend.index_nplike
                    and toindex.nplike is self._backend.index_nplike
                    and tooffsets.nplike is self._backend.index_nplike
                    and self._tags.nplike is self._backend.index_nplike
                    and self._index.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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
                        totags,
                        toindex,
                        contents,
                        parameters=self._parameters,
                    ),
                )

            else:
                offsets = ak.index.Index64.zeros(
                    0,
                    nplike=self._backend.index_nplike,
                    dtype=np.int64,
                )
                return (
                    offsets,
                    UnionArray(
                        self._tags,
                        self._index,
                        contents,
                        parameters=self._parameters,
                    ),
                )

    def _mergeable_next(self, other, mergebool):
        return True

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "to merge this array with 'others', at least one other must be provided"
                )
            )

        head = [self]
        tail = []

        for i in range(len(others)):
            head.append(others[i])

        if any(
            isinstance(x.backend.nplike, ak._typetracer.TypeTracer) for x in head + tail
        ):
            head = [
                x
                if isinstance(x.backend.nplike, ak._typetracer.TypeTracer)
                else x.to_typetracer()
                for x in head
            ]
            tail = [
                x
                if isinstance(x.backend.nplike, ak._typetracer.TypeTracer)
                else x.to_typetracer()
                for x in tail
            ]

        return (head, tail)

    def _reverse_merge(self, other):
        theirlength = other.length
        mylength = self.length

        tags = ak.index.Index8.empty(
            theirlength + mylength, nplike=self._backend.index_nplike
        )
        index = ak.index.Index64.empty(
            theirlength + mylength, nplike=self._backend.index_nplike
        )

        contents = [other]
        contents.extend(self.contents)

        assert tags.nplike is self._backend.index_nplike
        self._handle_error(
            self._backend["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data,
                0,
                theirlength,
                0,
            )
        )

        assert index.nplike is self._backend.index_nplike
        self._handle_error(
            self._backend["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data,
                0,
                theirlength,
            )
        )

        assert (
            tags.nplike is self._backend.index_nplike
            and self.tags.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
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

        assert (
            index.nplike is self._backend.index_nplike
            and self.index.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
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
            raise ak._errors.wrap_error(
                AssertionError("FIXME: handle UnionArray with more than 127 contents")
            )

        parameters = ak._util.merge_parameters(
            self._parameters,
            other._parameters,
            exclude=ak._util.meaningful_parameters,
        )

        return ak.contents.UnionArray.simplified(
            tags, index, contents, parameters=parameters
        )

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        nexttags = ak.index.Index8.empty(
            total_length, nplike=self._backend.index_nplike
        )
        nextindex = ak.index.Index64.empty(
            total_length, nplike=self._backend.index_nplike
        )

        nextcontents = []
        length_so_far = 0

        parameters = self._parameters
        for array in head:
            parameters = ak._util.merge_parameters(parameters, array._parameters, True)
            if isinstance(array, ak.contents.UnionArray):
                union_tags = ak.index.Index(array.tags)
                union_index = ak.index.Index(array.index)
                union_contents = array.contents
                assert (
                    nexttags.nplike is self._backend.index_nplike
                    and union_tags.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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
                    nextindex.nplike is self._backend.index_nplike
                    and union_index.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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

            elif isinstance(array, ak.contents.EmptyArray):
                pass

            else:
                assert nexttags.nplike is self._backend.index_nplike
                self._handle_error(
                    self._backend[
                        "awkward_UnionArray_filltags_const",
                        nexttags.dtype.type,
                    ](
                        nexttags.data,
                        length_so_far,
                        array.length,
                        len(nextcontents),
                    )
                )

                assert nextindex.nplike is self._backend.index_nplike
                self._handle_error(
                    self._backend[
                        "awkward_UnionArray_fillindex_count", nextindex.dtype.type
                    ](nextindex.data, length_so_far, array.length)
                )

                length_so_far += array.length
                nextcontents.append(array)

        if len(nextcontents) > 127:
            raise ak._errors.wrap_error(
                ValueError("FIXME: handle UnionArray with more than 127 contents")
            )

        next = ak.contents.UnionArray.simplified(
            nexttags,
            nextindex,
            nextcontents,
            parameters=parameters,
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        contents = []
        for content in self._contents:
            contents.append(content._fill_none(value))
        return UnionArray.simplified(
            self._tags,
            self._index,
            contents,
            parameters=self._parameters,
        )

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._local_index(axis, depth))
            return UnionArray(
                self._tags,
                self._index,
                contents,
                parameters=self._parameters,
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            contents = []
            for content in self._contents:
                contents.append(
                    content._combinations(
                        n, replacement, recordlookup, parameters, axis, depth
                    )
                )
            return ak.unionarray.UnionArray(
                self._tags,
                self._index,
                contents,
                parameters=self._parameters,
            )

    def _numbers_to_type(self, name):
        contents = []
        for x in self._contents:
            contents.append(x._numbers_to_type(name))
        return ak.contents.UnionArray(
            self._tags,
            self._index,
            contents,
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            merge=True,
            mergebool=True,
        )
        if isinstance(simplified, ak.contents.UnionArray):
            raise ak._errors.wrap_error(
                ValueError("cannot check if an irreducible UnionArray is unique")
            )

        return simplified._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            merge=True,
            mergebool=True,
        )
        if isinstance(simplified, ak.contents.UnionArray):
            raise ak._errors.wrap_error(
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
        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            mergebool=True,
        )
        if simplified.length == 0:
            return ak.contents.NumpyArray(
                self._backend.nplike.empty(0, np.int64),
                parameters=None,
                backend=self._backend,
            )

        if isinstance(simplified, ak.contents.UnionArray):
            raise ak._errors.wrap_error(
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

        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            mergebool=True,
        )
        if simplified.length == 0:
            return simplified

        if isinstance(simplified, ak.contents.UnionArray):
            raise ak._errors.wrap_error(
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
        # If we have a UnionArray, it must be irreducible, thanks to the
        # canonical checks in the constructor.
        raise ak._errors.wrap_error(
            ValueError(f"cannot call ak.{reducer.name} on an irreducible UnionArray")
        )

    def _validity_error(self, path):
        for i in range(len(self.contents)):
            if (
                self._backend.nplike.known_shape
                and self.index.length < self.tags.length
            ):
                return f'at {path} ("{type(self)}"): len(index) < len(tags)'

            lencontents = self._backend.index_nplike.empty(
                len(self.contents), dtype=np.int64
            )
            if self._backend.nplike.known_shape:
                for i in range(len(self.contents)):
                    lencontents[i] = self.contents[i].length

            error = self._backend[
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
                sub = self.contents[i]._validity_error(path + f".content({i})")
                if sub != "":
                    return sub

            return ""

    def _nbytes_part(self):
        result = self.tags._nbytes_part() + self.index._nbytes_part()
        for content in self.contents:
            result = result + content._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._pad_none(target, axis, depth, clip))
            return ak.contents.UnionArray.simplified(
                self.tags,
                self.index,
                contents,
                parameters=self._parameters,
            )

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

            this_length = 0
            if len(this_index) != 0:
                this_length = this_index.max() + 1

            values.append(
                content._to_arrow(
                    pyarrow, mask_node, this_validbytes, this_length, options
                )
            )

        types = pyarrow.union(
            [
                pyarrow.field(str(i), values[i].type).with_nullable(
                    mask_node is not None or self._contents[i].is_option
                )
                for i in range(len(values))
            ],
            "dense",
            list(range(len(values))),
        )

        if not issubclass(npindex.dtype.type, np.int32):
            npindex = npindex.astype(np.int32)

        return pyarrow.Array.from_buffers(
            ak._connect.pyarrow.to_awkwardarrow_type(
                types,
                options["extensionarray"],
                options["record_is_scalar"],
                None,
                self,
            ),
            nptags.shape[0],
            [
                None,
                ak._connect.pyarrow.to_length(nptags, length),
                ak._connect.pyarrow.to_length(npindex, length),
            ],
            children=values,
        )

    def _to_numpy(self, allow_missing):
        contents = [
            self.project(i)._to_numpy(allow_missing) for i in range(len(self.contents))
        ]

        if any(isinstance(x, self._backend.nplike.ma.MaskedArray) for x in contents):
            try:
                out = self._backend.nplike.ma.concatenate(contents)
            except Exception as err:
                raise ak._errors.wrap_error(
                    ValueError(f"cannot convert {self} into numpy.ma.MaskedArray")
                ) from err
        else:
            try:
                out = numpy.concatenate(contents)
            except Exception as err:
                raise ak._errors.wrap_error(
                    ValueError(f"cannot convert {self} into np.ndarray")
                ) from err

        tags = numpy.asarray(self.tags)
        for tag, content in enumerate(contents):
            mask = tags == tag
            out[mask] = content
        return out

    def _completely_flatten(self, backend, options):
        out = []
        for i in range(len(self._contents)):
            index = self._index[self._tags.data == i]
            out.extend(
                self._contents[i]
                ._carry(index, False)
                ._completely_flatten(backend, options)
            )
        return out

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:
            if options["return_simplified"]:
                make = UnionArray.simplified
            else:
                make = UnionArray

            def continuation():
                return make(
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
                    parameters=self._parameters if options["keep_parameters"] else None,
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
            backend=self._backend,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._errors.wrap_error(AssertionError(result))

    def to_packed(self) -> Self:
        tags = self._tags.raw(self._backend.nplike)
        original_index = index = self._index.raw(self._backend.nplike)[: tags.shape[0]]

        contents = list(self._contents)

        for tag in range(len(self._contents)):
            is_tag = tags == tag
            num_tag = self._backend.index_nplike.count_nonzero(is_tag)

            if len(contents[tag]) > num_tag:
                if original_index is index:
                    index = index.copy()
                index[is_tag] = self._backend.index_nplike.arange(
                    num_tag, dtype=index.dtype
                )
                contents[tag] = self.project(tag)

            contents[tag] = contents[tag].to_packed()

        return UnionArray(
            ak.index.Index8(tags, nplike=self._backend.index_nplike),
            ak.index.Index(index, nplike=self._backend.index_nplike),
            contents,
            parameters=self._parameters,
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

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        tags = self._tags.to_nplike(backend.index_nplike)
        index = self._index.to_nplike(backend.index_nplike)
        contents = [content.to_backend(backend) for content in self._contents]
        return UnionArray(
            tags,
            index,
            contents,
            parameters=self._parameters,
        )

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return (
            self.tags == other.tags
            and self.index.is_equal_to(other.index, index_dtype, numpyarray)
            and len(self.contents) == len(other.contents)
            and all(
                [
                    self.contents[i].is_equal_to(
                        other.contents[i], index_dtype, numpyarray
                    )
                    for i in range(len(self.contents))
                ]
            )
        )
