# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Content(object):
    def __getitem__(self, where):
        if ak._util.isint(where):
            return self._getitem_at(where)

        elif isinstance(where, slice) and where.step is None:
            return self._getitem_range(where)

        elif isinstance(where, slice):
            raise NotImplementedError("needs _getitem_next")

        elif ak._util.isstr(where):
            return self._getitem_field(where)

        elif where is np.newaxis:
            raise NotImplementedError("needs _getitem_next")

        elif where is Ellipsis:
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, tuple):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, ak.highlevel.Array):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, Content):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, Iterable) and all(ak._util.isstr(x) for x in where):
            return self._getitem_fields(where)

        elif isinstance(where, Iterable):
            raise NotImplementedError("needs _getitem_next")

        else:
            raise TypeError(
                "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                "integer/boolean arrays (possibly with variable-length nested "
                "lists or missing values), field name (str) or names (non-tuple "
                "iterable of str) are valid indices for slicing, not\n\n    "
                + repr(where)
            )
