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

    def _handle_error(self, error, slicer=None):
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")

            message = error.str.decode(errors="surrogateescape")

            if error.pass_through:
                raise ValueError(message + filename)

            else:
                if error.id != ak._util.kSliceNone and self._identifier is not None:
                    # FIXME https://github.com/scikit-hep/awkward-1.0/blob/45d59ef4ae45eebb02995b8e1acaac0d46fb9573/src/libawkward/util.cpp#L443-L450
                    pass

                if error.attempt != ak._util.kSliceNone:
                    message += " while attempting to get index {0}".format(
                        error.attempt
                    )

                message += filename

                if slicer is None:
                    raise ValueError(message)
                else:
                    raise NestedIndexError(self, slicer, message)

    def _headtail(self, oldtail):
        if len(oldtail) == 0:
            return (), ()
        else:
            return oldtail[0], oldtail[1:]

    def _getitem_next_field(self, head, tail, advanced):
        nexthead, nexttail = self._headtail(tail)
        return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

    def _getitem_next_fields(self, head, tail, advanced):
        only_fields, not_fields = [], []
        for x in tail:
            if ak._util.isstr(x) or isinstance(x, list):
                only_fields.append(x)
            else:
                not_fields.append(x)
        nexthead, nexttail = self._headtail(tuple(not_fields))
        return self._getitem_fields(head, tuple(only_fields))._getitem_next(
            nexthead, nexttail, advanced
        )

    def _getitem_next_newaxis(self, tail, advanced):
        nexthead, nexttail = self._headtail(tail)
        return ak._v2.contents.RegularArray(
            self._getitem_next(nexthead, nexttail, advanced),
            1,  # size
            0,  # zeros_length is irrelevant when the size is 1 (!= 0)
            None,
            None,
        )

    def __getitem__(self, where):
        try:
            if ak._util.isint(where):
                return self._getitem_at(where)

            elif isinstance(where, slice) and where.step is None:
                return self._getitem_range(where)

            elif isinstance(where, slice):
                return self.__getitem__((where,))

            elif ak._util.isstr(where):
                return self._getitem_field(where)

            elif where is np.newaxis:
                return self.__getitem__((where,))

            elif where is Ellipsis:
                return self.__getitem__((where,))

            elif isinstance(where, tuple):
                if len(where) == 0:
                    return self
                nextwhere = tuple(self._prepare_tuple_item(x) for x in where)
                next = ak._v2.contents.RegularArray(self, len(self), 1)
                out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
                if len(out) == 0:
                    raise NotImplementedError
                else:
                    return out._getitem_at(0)

            elif isinstance(where, ak.highlevel.Array):
                return self.__getitem__(where.layout)

            elif isinstance(where, ak.layout.Content):
                return self.__getitem__(v1_to_v2(where))

            elif isinstance(where, ak._v2.contents.numpyarray.NumpyArray):
                carry, allow_lazy = self._prepare_array(where)
                carried = self._carry_asrange(carry)
                if carried is None:
                    carried = self._carry(carry, allow_lazy, NestedIndexError)
                return _getitem_ensure_shape(carried, where.shape)

            elif isinstance(where, Content):
                return self.__getitem__((where,))

            elif isinstance(where, Iterable) and all(ak._util.isstr(x) for x in where):
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
                    + repr(where).replace("\n", "\n    ")
                )

        except NestedIndexError as err:

            def format_slice(x):
                if isinstance(x, slice):
                    if x.step is None:
                        return "{0}:{1}".format(
                            "" if x.start is None else x.start,
                            "" if x.stop is None else x.stop,
                        )
                    else:
                        return "{0}:{1}:{2}".format(
                            "" if x.start is None else x.start,
                            "" if x.stop is None else x.stop,
                            x.step,
                        )
                elif isinstance(x, tuple):
                    return "(" + ", ".join(format_slice(y) for y in x) + ")"
                elif isinstance(x, ak._v2.index.Index64):
                    return str(x.data)
                elif isinstance(x, Content):
                    return str(ak.Array(v2_to_v1(x)))
                else:
                    return repr(x)

            raise IndexError(
                """cannot slice

    {0}

with

    {1}

at inner {2} of length {3}, using sub-slice {4}.{5}""".format(
                    repr(ak.Array(v2_to_v1(self))),
                    format_slice(where),
                    type(err.array).__name__,
                    len(err.array),
                    format_slice(err.slicer),
                    ""
                    if err.details is None
                    else "\n\n{0} error: {1}.".format(
                        type(err.array).__name__, err.details
                    ),
                )
            )

    def _prepare_tuple_item(self, item):
        if ak._util.isint(item):
            return int(item)

        elif isinstance(item, slice):
            return item

        elif ak._util.isstr(item):
            return item

        elif item is np.newaxis:
            return item

        elif item is Ellipsis:
            return item

        elif isinstance(item, ak.highlevel.Array):
            return self._prepare_tuple_item(item.layout)

        elif isinstance(item, ak.layout.Content):
            return self._prepare_tuple_item(v1_to_v2(item))

        elif isinstance(item, ak._v2.contents.NumpyArray):
            return self._prepare_array(item)[0]

        elif isinstance(
            item,
            (
                ak._v2.contents.ListOffsetArray,
                ak._v2.contents.ListArray,
                ak._v2.contents.RegularArray,
            ),
        ):
            return item.toListOffsetArray64(False)

        elif isinstance(
            item,
            (
                ak._v2.contents.IndexedOptionArray,
                ak._v2.contents.ByteMaskedArray,
                ak._v2.contents.BitMaskedArray,
                ak._v2.contents.UnmaskedArray,
            ),
        ):
            return item.toIndexedOptionArray64()

        elif isinstance(item, Iterable) and all(ak._util.isstr(x) for x in item):
            return list(item)

        elif isinstance(item, Iterable):
            return self._prepare_tuple_item(
                v1_to_v2(ak.operations.convert.to_layout(item))
            )

        else:
            raise TypeError(
                "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                "integer/boolean arrays (possibly with variable-length nested "
                "lists or missing values), field name (str) or names (non-tuple "
                "iterable of str) are valid indices for slicing, not\n\n    "
                + repr(item).replace("\n", "\n    ")
            )

    def _prepare_array(self, where):
        if issubclass(where.dtype.type, np.int64):
            carry = ak._v2.index.Index64(where.data.reshape(-1))
            allow_lazy = True
        elif issubclass(where.dtype.type, np.integer):
            carry = ak._v2.index.Index64(where.data.astype(np.int64).reshape(-1))
            allow_lazy = "copied"  # True, but also can be modified in-place
        elif issubclass(where.dtype.type, (np.bool_, bool)):
            carry = ak._v2.index.Index64(np.nonzero(where.data.reshape(-1))[0])
            allow_lazy = "copied"  # True, but also can be modified in-place
        else:
            raise TypeError(
                "one-dimensional array slice must be an array of integers or "
                "booleans, not\n\n    {0}".format(
                    repr(where.data).replace("\n", "\n    ")
                )
            )
        return carry, allow_lazy

    def _carry_asrange(self, carry):
        assert isinstance(carry, ak._v2.index.Index)

        result = carry.nplike.empty(1, dtype=np.bool_)
        self._handle_error(
            carry.nplike[
                "awkward_Index_iscontiguous",  # badly named
                np.bool_,
                carry.dtype.type,
            ](
                result,
                carry.to(carry.nplike),
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

    def _range_identifier(self, start, stop):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _field_identifier(self, field):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _fields_identifier(self, fields):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError

    def _carry_identifier(self, carry, exception):
        if self._identifier is None:
            return None
        else:
            raise NotImplementedError


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
        return "cannot slice {0} with {1}{2}".format(
            type(self._array).__name__,
            repr(self._slicer),
            "" if self._details is None else ": " + self._details,
        )
