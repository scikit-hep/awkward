# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numbers

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Content(object):
    def _init(self, identifier, parameters):
        if parameters is None:
            parameters = {}

        if identifier is not None and not isinstance(
            identifier, ak._v2.identifier.Identifier
        ):
            raise TypeError(
                "{0} 'identifier' must be an Identifier or None, not {1}".format(
                    type(self).__name__, repr(identifier)
                )
            )
        if not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be a dict or None, not {1}".format(
                    type(self).__name__, repr(identifier)
                )
            )

        self._identifier = identifier
        self._parameters = parameters

    @property
    def identifier(self):
        return self._identifier

    @property
    def parameters(self):
        return self._parameters

    def __getitem__(self, where):
        if isinstance(where, numbers.Integral):
            return self._getitem_at(where)

        elif isinstance(where, slice) and where.step is None:
            return self._getitem_range(where)

        elif isinstance(where, slice):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, str):
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

        elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
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
