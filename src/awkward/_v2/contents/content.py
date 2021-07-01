# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.tmp_for_testing import v1_to_v2

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

    def handle_error(self, error):
        if error.filename is None:
            filename = ""
        else:
            filename = error.filename.decode(errors="surrogateescape")

        message = error.str.decode(errors="surrogateescape")

        if error.pass_through:
            raise ValueError(message + filename)

        else:
            if error.id != ak._util.kSliceNone and self._identifier is not None:
                # FIXME https://github.com/scikit-hep/awkward-1.0/blob/45d59ef4ae45eebb02995b8e1acaac0d46fb9573/src/libawkward/util.cpp#L443-L450
                pass

            if error.attempt != ak._util.kSliceNone:
                message += " (attempting to get {0})".format(error.attempt)

            # FIXME: attempt to pretty-print the array at this point; fall back to class name
            pretty = " in " + type(self).__name__

            message += pretty
            raise ValueError(message + filename)

    def __getitem__(self, where):
        import awkward._v2.contents.numpyarray

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

        elif isinstance(where, awkward._v2.contents.numpyarray.NumpyArray):
            if issubclass(where.data.dtype.type, (np.integer, np.bool_)):
                return self._getitem_array(where.data)
            else:
                raise TypeError(
                    "one-dimensional array slice must be an array of integers or "
                    "booleans, not\n\n    {0}".format(
                        repr(where.data).replace("\n", "\n    ")
                    )
                )

        elif isinstance(where, Content):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            return self._getitem_fields(where)

        elif isinstance(where, Iterable):
            return self.__getitem__(v1_to_v2(ak.operations.convert.to_layout(where)))

        else:
            raise TypeError(
                "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                "integer/boolean arrays (possibly with variable-length nested "
                "lists or missing values), field name (str) or names (non-tuple "
                "iterable of str) are valid indices for slicing, not\n\n    "
                + repr(where)
            )
