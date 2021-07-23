# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content, NestedIndexError

np = ak.nplike.NumpyMetadata.instance()


class EmptyArray(Content):
    def __init__(self, identifier=None, parameters=None):
        self._init(identifier, parameters)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        return indent + pre + "<EmptyArray len='0'/>" + post

    @property
    def form(self):
        return ak._v2.forms.EmptyForm(
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

    def _getitem_field(self, where):
        raise IndexError("field " + repr(where) + " not found")

    def _getitem_fields(self, where):
        raise IndexError("fields " + repr(where) + " not found")

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
            raise IndexError(
                "cannot slice "
                + type(self).__name__
                + "by a field name because it has no fields"
            )

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
