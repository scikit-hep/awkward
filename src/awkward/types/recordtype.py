# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import permutations

import awkward as ak
import awkward._prettyprint
from awkward._behavior import find_record_typestr
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._typing import Self, final
from awkward._util import UNSET
from awkward.types.type import Type


@final
class RecordType(Type):
    def copy(
        self,
        *,
        contents: list[Type] = UNSET,
        fields: list[str] | None = UNSET,
        parameters=UNSET,
    ) -> Self:
        return RecordType(
            self._contents if contents is UNSET else contents,
            self._fields if fields is UNSET else fields,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __init__(self, contents, fields, *, parameters=None):
        if not isinstance(contents, Iterable):
            raise TypeError(
                "{} 'contents' must be iterable, not {}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)
        for content in contents:
            if not isinstance(content, Type):
                raise TypeError(
                    "{} all 'contents' must be Type subclasses, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'fields' must be iterable, not {contents!r}"
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{} 'parameters' must be of type dict or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._contents = contents
        self._fields = fields
        self._parameters = parameters

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self):
        return self._fields

    @property
    def is_tuple(self):
        return self._fields is None

    _str_parameters_exclude = ("__categorical__", "__record__")

    def _str(self, indent, compact, behavior):
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
            name = self.parameter("__record__")

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

    def _is_equal_to(self, other, all_parameters: bool) -> bool:
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

            self_contents = [self.content(f) for f in self._fields]
            other_contents = [other.content(f) for f in other._fields]

            return any(
                all(
                    this._is_equal_to(that, all_parameters)
                    for this, that in zip(self_contents, contents)
                )
                for contents in permutations(other_contents)
            )
        # Mixed
        else:
            return False

    def index_to_field(self, index):
        return ak.forms.RecordForm.index_to_field(self, index)

    def field_to_index(self, field):
        return ak.forms.RecordForm.field_to_index(self, field)

    def has_field(self, field):
        return ak.forms.RecordForm.has_field(self, field)

    def content(self, index_or_field):
        return ak.forms.RecordForm.content(self, index_or_field)
