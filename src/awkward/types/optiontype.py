# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping

from awkward._behavior import find_array_typestr
from awkward._parameters import (
    parameters_are_equal,
    parameters_union,
    type_parameters_equal,
)
from awkward._typing import Any, JSONMapping, final
from awkward._util import UNSET, Sentinel
from awkward.types.listtype import ListType
from awkward.types.regulartype import RegularType
from awkward.types.type import Type
from awkward.types.uniontype import UnionType


@final
class OptionType(Type):
    def copy(
        self,
        *,
        content: Type | Sentinel = UNSET,
        parameters: JSONMapping | Sentinel | None = UNSET,
    ) -> OptionType:
        return OptionType(
            self._content if content is UNSET else content,  # type: ignore[arg-type]
            parameters=self._parameters if parameters is UNSET else parameters,  # type: ignore[arg-type]
        )

    def __init__(self, content: Type, *, parameters: JSONMapping | None = None):
        if not isinstance(content, Type):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Type subclass, not {content!r}"
            )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be of type Mapping or None, not {parameters!r}"
            )
        self._content: Type = content
        self._parameters: JSONMapping | None = parameters

    @property
    def content(self) -> Type:
        return self._content

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
        typestr = find_array_typestr(behavior, self._parameters)

        head = []
        tail = []
        if typestr is not None:
            content_out = [typestr]

        else:
            content_out = self._content._str(indent, compact, behavior)
            params = self._str_parameters()
            if params is None:
                if isinstance(
                    self._content, (RegularType, ListType)
                ) and self._content.parameter("__array__") not in (
                    "string",
                    "bytestring",
                    "char",
                    "byte",
                ):
                    head = ["option["]
                    tail = ["]"]
                else:
                    head = ["?"]

            else:
                head = ["option["]
                tail = [f", {params}]"]

        return [
            *head,
            self._str_categorical_begin(),
            *content_out,
            self._str_categorical_end(),
            *tail,
        ]

    def __repr__(self) -> str:
        args = [repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def simplify_option_union(self) -> Type:
        if isinstance(self._content, UnionType):
            contents = []
            for content in self._content.contents:
                if isinstance(content, OptionType):
                    contents.append(
                        OptionType(
                            content.content,
                            parameters=parameters_union(
                                self._parameters, content._parameters
                            ),
                        )
                    )

                else:
                    contents.append(OptionType(content, parameters=self._parameters))

            return UnionType(contents, parameters=self._content.parameters)

        else:
            return self

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and self._content._is_equal_to(
                other._content, all_parameters=all_parameters
            )
        )
