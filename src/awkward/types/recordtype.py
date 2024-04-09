# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

import awkward as ak
import awkward._prettyprint
from awkward._behavior import find_record_typestr
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import Any, JSONMapping, Self, cast, final
from awkward._util import UNSET, Sentinel
from awkward.errors import FieldNotFoundError
from awkward.types.type import Type


@final
class RecordType(Type):
    def copy(
        self,
        *,
        contents: Iterable[Type] | Sentinel = UNSET,
        fields: Iterable[str] | Sentinel | None = UNSET,
        parameters: JSONMapping | Sentinel | None = UNSET,
    ) -> Self:
        return RecordType(
            self._contents if contents is UNSET else contents,  # type: ignore[arg-type]
            self._fields if fields is UNSET else fields,  # type: ignore[arg-type]
            parameters=self._parameters if parameters is UNSET else parameters,  # type: ignore[arg-type]
        )

    def __init__(
        self,
        contents: Iterable[Type],
        fields: Iterable[str],
        *,
        parameters: JSONMapping | None = None,
    ):
        if not isinstance(contents, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'contents' must be iterable, not {contents!r}"
            )
        elif not isinstance(contents, list):
            contents = list(contents)

        for content in contents:
            if not isinstance(content, Type):
                raise TypeError(
                    f"{type(self).__name__} all 'contents' must be Type subclasses, not {content!r}"
                )
        if fields is not None:
            if not isinstance(fields, Iterable):
                raise TypeError(
                    f"{type(self).__name__} 'fields' must be iterable, not {contents!r}"
                )
            elif not isinstance(fields, list):
                fields = list(fields)

        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be of type Mapping or None, not {parameters!r}"
            )
        self._contents: list[Type] = contents
        self._fields: list[str] = fields
        self._parameters: JSONMapping | None = parameters

    @property
    def contents(self) -> list[Type]:
        return self._contents

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def is_tuple(self) -> bool:
        return self._fields is None

    _str_parameters_exclude: tuple[str, ...] = ("__categorical__", "__record__")

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
        typestr = find_record_typestr(behavior, self._parameters)
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

            params = self._str_parameters()
            name = cast("str | None", self.parameter("__record__"))

            if name is not None:
                if (
                    not ak._prettyprint.is_identifier.match(name)
                    or name
                    in (
                        "unknown",
                        "string",
                        "bytes",
                        "option",
                        "tuple",
                        "struct",
                        "union",
                        "categorical",
                    )
                    or name in ak.types.numpytype._primitive_to_dtype_dict
                ):
                    if params is None:
                        params = 'parameters={"__record__": ' + json.dumps(name) + "}"
                    else:
                        params = (
                            'parameters={"__record__": '
                            + json.dumps(name)
                            + ", "
                            + params[12:]
                        )
                    name = None

            if not self.is_tuple:
                pairs = []
                for k, v in zip(self._fields, children):
                    if ak._prettyprint.is_identifier.match(k) is None:
                        key_str = json.dumps(k)
                    else:
                        key_str = k
                    pairs.append([key_str, ": ", *v])
                flat_pairs = [y for x in pairs for y in x]

            if params is None:
                if self.is_tuple:
                    flat_children = [y for x in children for y in x]
                    if name is None:
                        out = ["(", pre, *flat_children, post, ")"]
                    else:
                        out = [name, "[", pre, *flat_children, post, "]"]
                else:
                    if name is None:
                        out = ["{", pre, *flat_pairs, post, "}"]
                    else:
                        out = [name, "[", pre, *flat_pairs, post, "]"]

            else:
                if self.is_tuple:
                    flat_children = [y for x in children for y in x]
                    if name is None:
                        out = ["tuple[[", pre, *flat_children, post, "], ", params, "]"]
                    else:
                        c = "" if len(self._contents) == 0 else ", "
                        out = [name, "[", pre, *flat_children, c, post, params, "]"]
                else:
                    if name is None:
                        out = ["struct[{", pre, *flat_pairs, post, "}, ", params, "]"]
                    else:
                        c = "" if len(self._contents) == 0 else ", "
                        out = [name, "[", pre, *flat_pairs, c, post, params, "]"]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        if not isinstance(other, type(self)):
            return False

        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        if not compare_parameters(self._parameters, other._parameters):
            return False

        # Both tuples
        if self.is_tuple and other.is_tuple:
            return all(
                this._is_equal_to(that, all_parameters)
                for this, that in zip(self._contents, other._contents)
            )
        # Both records
        elif not (self.is_tuple or other.is_tuple):
            if set(self._fields) != set(other._fields):
                return False

            return all(
                content._is_equal_to(other.content(field), all_parameters)
                for field, content in zip(self._fields, self._contents)
            )

        # Mixed
        else:
            return False

    def index_to_field(self, index: int) -> str:
        if 0 <= index < len(self._contents):
            if self._fields is None:
                return str(index)
            else:
                return self._fields[index]
        else:
            raise IndexError(
                f"no index {index} in record with {len(self._contents)} fields"
            )

    def field_to_index(self, field: str) -> int:
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._fields.index(field)
            except ValueError:
                pass
            else:
                return i
        raise FieldNotFoundError(
            f"no field {field!r} in record with {len(self._contents)} fields"
        )

    def has_field(self, field: str) -> bool:
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                return False
            else:
                return 0 <= i < len(self._contents)
        else:
            return field in self._fields

    def content(self, index_or_field: int | str) -> Type:
        if is_integer(index_or_field):
            index = int(index_or_field)
        elif isinstance(index_or_field, str):
            index = self.field_to_index(index_or_field)
        else:
            raise TypeError(
                f"index_or_field must be an integer (index) or string (field), not {index_or_field!r}"
            )
        return self._contents[index]
