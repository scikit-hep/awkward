# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import permutations

from awkward._behavior import find_array_typestr
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._typing import Any, JSONMapping, Self, final
from awkward._util import UNSET, Sentinel
from awkward.types.type import Type


@final
class UnionType(Type):
    _contents: list[Type]

    def copy(
        self,
        *,
        contents: list[Type] | Sentinel = UNSET,
        parameters: JSONMapping | None | Sentinel = UNSET,
    ) -> Self:
        return UnionType(
            self._contents if contents is UNSET else contents,  # type: ignore[arg-type]
            parameters=self._parameters if parameters is UNSET else parameters,  # type: ignore[arg-type]
        )

    def __init__(
        self, contents: Iterable[Type], *, parameters: JSONMapping | None = None
    ):
        if not isinstance(contents, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'contents' must be iterable, not {contents!r}"
            )
        if not isinstance(contents, list):
            contents = list(contents)
        for content in contents:
            if not isinstance(content, Type):
                raise TypeError(
                    f"{type(self).__name__} all 'contents' must be Type subclasses, not {content!r}"
                )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be of type Mapping or None, not {parameters!r}"
            )
        self._contents: list[Type] = contents
        self._parameters = parameters

    @property
    def contents(self) -> list[Type]:
        return self._contents

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
        typestr = find_array_typestr(behavior, self._parameters)
        if typestr is not None:
            out = [typestr]

        else:
            if compact:
                pre, post = "", ""
            else:
                pre, post = "\n" + indent + "    ", "\n" + indent

            children = []
            for i, x in enumerate(self._contents):
                if i + 1 < len(self._contents):
                    if compact:
                        y = [*x._str(indent, compact, behavior), ", "]
                    else:
                        y = [
                            *x._str(indent + "    ", compact, behavior),
                            ",\n",
                            indent,
                            "    ",
                        ]
                else:
                    if compact:
                        y = x._str(indent, compact, behavior)
                    else:
                        y = x._str(indent + "    ", compact, behavior)
                children.append(y)

            flat_children = [y for x in children for y in x]
            params = self._str_parameters()

            if params is None:
                out = ["union[", pre, *flat_children, post, "]"]
            else:
                out = ["union[", pre, *flat_children, ", ", post, params, "]"]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self) -> str:
        args = [repr(self._contents), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and len(self._contents) == len(other._contents)
            and any(
                all(
                    this._is_equal_to(that, all_parameters)
                    for this, that in zip(self._contents, contents)
                )
                for contents in permutations(other._contents)
            )
        )
