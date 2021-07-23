# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content, NestedIndexError
from awkward._v2.forms.emptyform import EmptyForm

np = ak.nplike.NumpyMetadata.instance()


class EmptyArray(Content):
    def __init__(self, identifier=None, parameters=None):
        self._init(identifier, parameters)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        return indent + pre + "<EmptyArray len='0'/>" + post

    Form = EmptyForm

    @property
    def form(self):
        return self.Form(
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def nplike(self):
        return ak.nplike.Numpy.instance()

    def __len__(self):
        return 0

    def _getitem_at(self, where):
        raise NestedIndexError(self, where)

    def _getitem_range(self, where):
        return self

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        if len(carry) == 0:
            return self
        else:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data)
            else:
                raise exception("index out of range")

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike  # noqa: F841

        if head == ():
            raise NotImplementedError

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
            raise NotImplementedError

        elif isinstance(head, ak._v2.index.Index64):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NotImplementedError

        else:
            raise AssertionError(repr(head))
