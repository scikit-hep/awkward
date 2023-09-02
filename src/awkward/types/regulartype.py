# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from awkward._behavior import find_array_typestr
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import Self, final
from awkward._util import UNSET
from awkward.types.type import Type


@final
class RegularType(Type):
    def copy(
        self, *, content: Type = UNSET, size: ShapeItem = UNSET, parameters=UNSET
    ) -> Self:
        return RegularType(
            self._content if content is UNSET else content,
            size=self._size if size is UNSET else size,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __init__(self, content, size, *, parameters=None):
        if not isinstance(content, Type):
            raise TypeError(
                "{} 'content' must be a Type subtype, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not (size is unknown_length or (is_integer(size) and size >= 0)):
            raise ValueError(
                "{} 'size' must be a non-negative int or None, not {}".format(
                    type(self).__name__, repr(size)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{} 'parameters' must be of type dict or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._content = content
        self._size = size
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    def _get_typestr(self, behavior) -> str | None:
        typestr = find_array_typestr(behavior, self._parameters)
        if typestr is not None:
            return typestr

        if self._parameters is None:
            return None

        name = self._parameters.get("__array__")
        if name == "string":
            return "string"
        elif name == "bytestring":
            return "bytes"
        else:
            return None

    def _str(self, indent, compact, behavior):
        typestr = self._get_typestr(behavior)
        if typestr is not None:
            out = [f"{typestr}[{self._size}]"]

        else:
            params = self._str_parameters()

            if params is None:
                out = [
                    str(self._size),
                    " * ",
                    *self._content._str(indent, compact, behavior),
                ]
            else:
                out = [
                    "[",
                    str(self._size),
                    " * ",
                    *self._content._str(indent, compact, behavior),
                    ", ",
                    params,
                    "]",
                ]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._content), repr(self._size), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and (self._size == other._size)
            and self._content._is_equal_to(other._content, all_parameters)
        )
