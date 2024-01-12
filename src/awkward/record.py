# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Iterable

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import (
    register_backend_lookup_factory,
    regularize_backend,
)
from awkward._behavior import get_record_class
from awkward._layout import wrap_layout
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._regularize import is_integer
from awkward._typing import JSONSerializable, Self
from awkward._util import UNSET
from awkward.contents.content import Content

np = NumpyMetadata.instance()


class Record:
    """
    Represents a single value from a #ak.contents.RecordArray.

    As this is a columnar representation, the Record contains a
    #ak.layout.RecordArray, rather than the other way around.
    Its two fields are

    * `array`: the #ak.layout.RecordArray and
    * `at`: the index posiion where this Record is found.

    The Record shares a reference with its #ak.layout.RecordArray;
    it is not a copy.
    """

    def __init__(self, array, at):
        if not isinstance(array, ak.contents.RecordArray):
            raise TypeError(f"Record 'array' must be a RecordArray, not {array!r}")
        if not is_integer(at):
            raise TypeError(f"Record 'at' must be an integer, not {array!r}")
        if not (array.length is unknown_length or 0 <= at < array.length):
            raise ValueError(
                f"Record 'at' must be >= 0 and < len(array) == {array.length}, not {at}"
            )
        else:
            self._array = array
            self._at = at

    @property
    def array(self):
        return self._array

    @property
    def backend(self):
        return self._array.backend

    @property
    def at(self):
        return self._at

    @property
    def fields(self):
        return self._array.fields

    @property
    def is_tuple(self):
        return self._array.is_tuple

    def to_tuple(self) -> Self:
        return Record(self._array.to_tuple(), self._at)

    @property
    def contents(self):
        out = []
        for field in self._array.fields:
            out.append(self._array[field][self._at])
        return out

    def content(self, index_or_field):
        return self._array.content(index_or_field)[self._at]

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Record at="]
        out.append(repr(str(self._at)))
        out.append(">\n")
        out.append(self._array._repr(indent + "    ", "<array>", "</array>\n"))
        out.append(indent)
        out.append("</Record>")
        out.append(post)
        return "".join(out)

    def validity_error(self, path="layout.array"):
        return ak._do.validity_error(self._array, path)

    @property
    def parameters(self):
        return self._array.parameters

    def parameter(self, key):
        return self._array.parameter(key)

    def purelist_parameter(self, key) -> JSONSerializable:
        return self._array.purelist_parameters(key)

    def purelist_parameters(self, *keys):
        return self._array.purelist_parameters(*keys)

    @property
    def purelist_isregular(self):
        return self._array.purelist_isregular

    @property
    def purelist_depth(self):
        return 0

    @property
    def minmax_depth(self):
        mindepth, maxdepth = self._array.minmax_depth
        return mindepth - 1, maxdepth - 1

    @property
    def branch_depth(self):
        branch, depth = self._array.branch_depth
        return branch, depth - 1

    def _touch_data(self, recursive):
        self._array._touch_data(recursive)

    def _touch_shape(self, recursive):
        self._array._touch_shape(recursive)

    def __getitem__(self, where):
        return self._getitem(where)

    def _getitem(self, where):
        if is_integer(where):
            raise IndexError("scalar Record cannot be sliced by an integer")

        elif isinstance(where, slice):
            raise IndexError("scalar Record cannot be sliced by a range slice (`:`)")

        elif isinstance(where, str):
            return self._getitem_field(where)

        elif where is np.newaxis:
            raise IndexError("scalar Record cannot be sliced by np.newaxis (`None`)")

        elif where is Ellipsis:
            raise IndexError("scalar Record cannot be sliced by an ellipsis (`...`)")

        elif isinstance(where, tuple) and len(where) == 0:
            return self

        elif isinstance(where, tuple) and len(where) == 1:
            return self._getitem(where[0])

        elif isinstance(where, tuple) and isinstance(where[0], str):
            return self._getitem_field(where[0])._getitem(where[1:])

        elif isinstance(where, ak.highlevel.Array):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, ak.contents.Content):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Content):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            return self._getitem_fields(where)

        elif isinstance(where, Iterable):
            raise IndexError("scalar Record cannot be sliced by an array")

        else:
            raise TypeError(
                "only field name (str) or names (non-tuple iterable of str) "
                "are valid indices for slicing a scalar record, not\n\n    "
                + repr(where)
            )

    def _getitem_field(self, where, only_fields: tuple[str, ...] = ()) -> Content:
        return self._array._getitem_field(where)._getitem_at(self._at)

    def _getitem_fields(self, where, only_fields: tuple[str, ...] = ()):
        return self._array._getitem_fields(where)._getitem_at(self._at)

    def to_packed(self, recursive: bool = True) -> Self:
        if self._array.length == 1:
            return Record(self._array.to_packed(recursive), self._at)
        else:
            return Record(self._array[self._at : self._at + 1].to_packed(recursive), 0)

    def to_list(self, behavior=None):
        return self._to_list(behavior, None)

    def _to_list(self, behavior, json_conversions):
        overloaded = (
            get_record_class(self._array, behavior).__getitem__
            is not ak.highlevel.Record.__getitem__
        )

        if overloaded:
            record = wrap_layout(self, behavior=behavior)
            contents = []
            for field in self._array.fields:
                contents.append(record[field])
                if isinstance(contents[-1], (ak.highlevel.Array, ak.highlevel.Record)):
                    contents[-1] = contents[-1]._layout._to_list(
                        behavior, json_conversions
                    )
                elif hasattr(contents[-1], "tolist"):
                    contents[-1] = contents[-1].tolist()

        else:
            contents = [
                content[self._at : self._at + 1]._to_list(behavior, json_conversions)[0]
                for content in self._array.contents
            ]

        if self.is_tuple and json_conversions is None:
            return tuple(contents)
        else:
            return dict(zip(self._array.fields, contents))

    def to_backend(self, backend: Backend | str | None = None) -> Self:
        if backend is None:
            backend = self._array.backend
        else:
            backend = regularize_backend(backend)
        if backend is self._array.backend:
            return self
        else:
            return Record(self._array._to_backend(backend), self._at)

    def __copy__(self) -> Self:
        return self.copy()

    def __deepcopy__(self, memo):
        return Record(copy.deepcopy(self._array, memo), self._at)

    def copy(self, array=UNSET, at=UNSET) -> Self:
        return Record(
            self._array if array is UNSET else array, self._at if at is UNSET else at
        )


@register_backend_lookup_factory
def find_record_backend(obj: type):
    if issubclass(obj, Record):

        def finder(obj: Record):
            return obj.backend

        return finder
