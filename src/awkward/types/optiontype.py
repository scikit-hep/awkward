# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from awkward._behavior import find_array_typestr
from awkward._parameters import (
    parameters_are_equal,
    parameters_union,
    type_parameters_equal,
)
from awkward._typing import Self, final
from awkward._util import UNSET
from awkward.types.listtype import ListType
from awkward.types.regulartype import RegularType
from awkward.types.type import Type
from awkward.types.uniontype import UnionType


@final
class OptionType(Type):
    def copy(self, *, content: Type = UNSET, parameters=UNSET) -> Self:
        return OptionType(
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __init__(self, content, *, parameters=None):
        if not isinstance(content, Type):
            raise TypeError(
                "{} 'content' must be a Type subclass, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{} 'parameters' must be of type dict or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._content = content
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def _str(self, indent, compact, behavior):
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

    def __repr__(self):
        args = [repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def simplify_option_union(self):
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

    def _is_equal_to(self, other, all_parameters: bool) -> bool:
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
