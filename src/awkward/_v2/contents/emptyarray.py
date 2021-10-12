# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.emptyform import EmptyForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class EmptyArray(Content):
    def __init__(self, identifier=None, parameters=None):
        self._init(identifier, parameters)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        extra = self._repr_extra(indent + "    ")
        if len(extra) == 0:
            return indent + pre + "<EmptyArray len='0'/>" + post
        else:
            out = [indent, pre, "<EmptyArray len='0'>"]
            out.extend(extra)
            out.append("\n" + indent + "</EmptyArray>")
            out.append(post)
            return "".join(out)

    Form = EmptyForm

    @property
    def form(self):
        return self.Form(
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        return EmptyArray(self._typetracer_identifier(), self._parameters)

    @property
    def nplike(self):
        return ak.nplike.Numpy.instance()

    @property
    def nonvirtual_nplike(self):
        return None

    def __len__(self):
        return 0

    def toNumpyArray(self, dtype, nplike=None):
        return ak._v2.contents.numpyarray.NumpyArray(
            numpy.empty(0, dtype), self._identifier, self._parameters, nplike=nplike
        )

    def _getitem_nothing(self):
        return self

    def _getitem_at(self, where):
        raise NestedIndexError(self, where, "array is empty")

    def _getitem_range(self, where):
        return self

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        if len(carry) == 0:
            return self
        else:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, "array is empty")
            else:
                raise exception("array is empty")

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            raise NestedIndexError(self, head, "array is empty")

        elif isinstance(head, slice):
            raise NestedIndexError(self, head, "array is empty")

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            raise NestedIndexError(self, head, "array is empty")

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NestedIndexError(self, head, "array is empty")

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NestedIndexError(self, head, "array is empty")

        else:
            raise AssertionError(repr(head))

    def _localindex(self, axis, depth):
        return ak._v2.contents.numpyarray.NumpyArray(np.empty(0, np.int64))

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
        as_numpy = ak._v2.contents.NumpyArray(self)
        return as_numpy._argsort_next(
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
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        return self

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return ak._v2.contents.emptyarray.EmptyArray(self._identifier, self._parameters)

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
        as_numpy = self.toNumpyArray(reducer.preferred_dtype, nplike=parents.nplike)
        return as_numpy._reduce_next(
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
        return ""
