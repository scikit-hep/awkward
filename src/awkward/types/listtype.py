# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._typing import Self, final
from awkward._util import unset
from awkward.types.type import Type


@final
class ListType(Type):
    def copy(self, *, content: Type = unset, parameters=unset, typestr=unset) -> Self:
        return ListType(
            self._content if content is unset else content,
            parameters=self._parameters if parameters is unset else parameters,
            typestr=self._typestr if typestr is unset else typestr,
        )

    def __init__(self, content, *, parameters=None, typestr=None):
        if not isinstance(content, Type):
            raise TypeError(
                "{} 'content' must be a Type subtype, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{} 'parameters' must be of type dict or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise TypeError(
                "{} 'typestr' must be of type string or None, not {}".format(
                    type(self).__name__, repr(typestr)
                )
            )
        self._content = content
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    def _str(self, indent, compact):
        if self._typestr is not None:
            out = [self._typestr]

        elif self.parameter("__array__") == "string":
            out = ["string"]

        elif self.parameter("__array__") == "bytestring":
            out = ["bytes"]

        else:
            params = self._str_parameters()
            if params is None:
                out = ["var * ", *self._content._str(indent, compact)]
            else:
                out = ["[var * ", *self._content._str(indent, compact)] + [
                    f", {params}]"
                ]

        return [self._str_categorical_begin(), *out] + [self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other, all_parameters: bool):
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and self._content == other._content
        )
