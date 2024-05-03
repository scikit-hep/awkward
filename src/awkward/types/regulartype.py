# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping

from awkward._behavior import find_array_typestr
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import Any, JSONMapping, Self, cast, final
from awkward._util import UNSET, Sentinel
from awkward.types.type import Type


@final
class RegularType(Type):
    def copy(
        self,
        *,
        content: Type | Sentinel = UNSET,
        size: ShapeItem | Sentinel = UNSET,
        parameters: JSONMapping | Sentinel | None = UNSET,
    ) -> Self:
        return RegularType(
            self._content if content is UNSET else content,  # type: ignore[arg-type]
            size=self._size if size is UNSET else size,  # type: ignore[arg-type]
            parameters=self._parameters if parameters is UNSET else parameters,  # type: ignore[arg-type]
        )

    def __init__(
        self, content: Type, size: ShapeItem, *, parameters: JSONMapping | None = None
    ):
        if not isinstance(content, Type):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Type subtype, not {content!r}"
            )
        if not (size is unknown_length or (is_integer(size) and size >= 0)):
            raise ValueError(
                f"{type(self).__name__} 'size' must be a non-negative int or None, not {size!r}"
            )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be of type Mapping or None, not {parameters!r}"
            )
        self._content: Type = content
        self._size: ShapeItem = size
        self._parameters: JSONMapping | None = parameters

    @property
    def content(self) -> Type:
        return self._content

    @property
    def size(self) -> ShapeItem:
        return self._size

    def _get_typestr(self, behavior: Mapping | None) -> str | None:
        typestr = find_array_typestr(behavior, self._parameters)
        if typestr is not None:
            return typestr

        if self._parameters is None:
            return None

        name = cast("str | None", self._parameters.get("__array__"))
        if name == "string":
            return "string"
        elif name == "bytestring":
            return "bytes"
        else:
            return None

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
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

    def __repr__(self) -> str:
        args = [repr(self._content), repr(self._size), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and (self._size == other._size)
            and self._content._is_equal_to(other._content, all_parameters)
        )
