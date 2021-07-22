# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1

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
        if error.str is not None:
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

        try:
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
                if len(where) == 0:
                    return self
                next = ak._v2.contents.RegularArray(self, len(self), 1)
                return next._getitem_next(where[0], where[1:], None)

            elif isinstance(where, ak.highlevel.Array):
                raise NotImplementedError("needs _getitem_next")

            elif isinstance(where, awkward._v2.contents.numpyarray.NumpyArray):
                if issubclass(where.data.dtype.type, np.int64):
                    carry = ak._v2.index.Index64(where.data.reshape(-1))
                    allow_lazy = True
                elif issubclass(where.data.dtype.type, np.integer):
                    carry = ak._v2.index.Index64(
                        where.data.astype(np.int64).reshape(-1)
                    )
                    allow_lazy = "copied"  # True, but also can be modified in-place
                elif issubclass(where.data.dtype.type, (np.bool_, bool)):
                    carry = ak._v2.index.Index64(np.nonzero(where.data.reshape(-1))[0])
                    allow_lazy = "copied"  # True, but also can be modified in-place
                else:
                    raise TypeError(
                        "one-dimensional array slice must be an array of integers or "
                        "booleans, not\n\n    {0}".format(
                            repr(where.data).replace("\n", "\n    ")
                        )
                    )
                carried = self._carry_asrange(carry)
                if carried is None:
                    carried = self._carry(carry, allow_lazy=allow_lazy)
                return _getitem_ensure_shape(carried, where.data.shape)

            elif isinstance(where, Content):
                raise NotImplementedError("needs _getitem_next")

            elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
                return self._getitem_fields(where)

            elif isinstance(where, Iterable):
                return self.__getitem__(
                    v1_to_v2(ak.operations.convert.to_layout(where))
                )

            else:
                raise TypeError(
                    "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                    "integer/boolean arrays (possibly with variable-length nested "
                    "lists or missing values), field name (str) or names (non-tuple "
                    "iterable of str) are valid indices for slicing, not\n\n    "
                    + repr(where)
                )

        except NestedIndexError as err:
            if isinstance(where, Content):
                wherey = str(ak.Array(v2_to_v1(where)))
            else:
                wherey = repr(where)

            raise IndexError(
                """cannot slice

    {0}

with

    {1}

because an index is out of bounds (in {2} with length {3}, using sub-slice {4}{5})""".format(
                    repr(ak.Array(v2_to_v1(self))),
                    wherey,
                    type(err.array).__name__,
                    len(err.array),
                    repr(err.slicer),
                    "" if err.details is None else ", details: " + repr(err.details),
                )
            )

    def _carry_asrange(self, carry):
        assert isinstance(carry, ak._v2.index.Index)

        result = carry.nplike.empty(1, dtype=np.bool_)
        self.handle_error(
            carry.nplike[
                "awkward_Index_iscontiguous",  # badly named
                np.bool_,
                carry.dtype.type,
            ](
                result,
                carry.data,
                len(carry),
            )
        )
        if result[0]:
            if len(carry) == len(self):
                return self
            elif len(carry) < len(self):
                return self._getitem_range(slice(0, len(carry)))
            else:
                raise IndexError
        else:
            return None


def _getitem_ensure_shape(array, shape):
    assert isinstance(array, Content)
    if len(shape) == 1:
        assert len(array) == shape[0]
        return array
    else:
        return _getitem_ensure_shape(
            ak._v2.contents.RegularArray(array, shape[-1], shape[-2]), shape[:-1]
        )


class NestedIndexError(IndexError):
    def __init__(self, array, slicer, details=None):
        self._array = array
        self._slicer = slicer
        self._details = details

    @property
    def array(self):
        return self._array

    @property
    def slicer(self):
        return self._slicer

    @property
    def details(self):
        return self._details

    def __str__(self):
        return """cannot slice

    {0}

with

    {1}

because an index is out of bounds{2}""".format(
            repr(self._array),
            repr(self._slicer),
            "" if self._details is None else " (details: " + repr(self._details) + ")",
        )
