# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numbers


class Content(object):
    def __getitem__(self, where):
        if isinstance(where, numbers.Integral):
            return self._getitem_at(where)
        elif isinstance(where, slice) and where.step is None:
            return self._getitem_range(where)
        elif isinstance(where, str):
            return self._getitem_field(where)
        elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            return self._getitem_fields(where)
        else:
            raise AssertionError(where)
