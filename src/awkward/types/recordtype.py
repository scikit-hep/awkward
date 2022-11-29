# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
from collections.abc import Iterable

import awkward as ak
import awkward._prettyprint
from awkward.forms.form import _parameters_equal
from awkward.types.type import Type


class RecordType(Type):
    def __init__(self, contents, fields, *, parameters=None, typestr=None):
        if not isinstance(contents, Iterable):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)
        for content in contents:
            if not isinstance(content, Type):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} all 'contents' must be Type subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'fields' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'parameters' must be of type dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'typestr' must be of type string or None, not {}".format(
                        type(self).__name__, repr(typestr)
                    )
                )
            )
        self._contents = contents
        self._fields = fields
        self._parameters = parameters
        self._typestr = typestr

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

    def _str(self, indent, compact):
        if self._typestr is not None:
            out = [self._typestr]

        else:
            if compact:
                pre, post = "", ""
            else:
                pre, post = "\n" + indent + "    ", "\n" + indent

            children = []
            for i, x in enumerate(self._contents):
                if i + 1 < len(self._contents):
                    if compact:
                        y = x._str(indent, compact) + [", "]
                    else:
                        y = x._str(indent + "    ", compact) + [",\n", indent, "    "]
                else:
                    if compact:
                        y = x._str(indent, compact)
                    else:
                        y = x._str(indent + "    ", compact)
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
                    pairs.append([key_str, ": "] + v)
                flat_pairs = [y for x in pairs for y in x]

            if params is None:
                if self.is_tuple:
                    flat_children = [y for x in children for y in x]
                    if name is None:
                        out = ["(", pre] + flat_children + [post, ")"]
                    else:
                        out = [name, "[", pre] + flat_children + [post, "]"]
                else:
                    if name is None:
                        out = ["{", pre] + flat_pairs + [post, "}"]
                    else:
                        out = [name, "[", pre] + flat_pairs + [post, "]"]

            else:
                if self.is_tuple:
                    flat_children = [y for x in children for y in x]
                    if name is None:
                        out = (
                            ["tuple[[", pre]
                            + flat_children
                            + [post, "], ", params, "]"]
                        )
                    else:
                        c = "" if len(self._contents) == 0 else ", "
                        out = [name, "[", pre] + flat_children + [c, post, params, "]"]
                else:
                    if name is None:
                        out = (
                            ["struct[{", pre] + flat_pairs + [post, "}, ", params, "]"]
                        )
                    else:
                        c = "" if len(self._contents) == 0 else ", "
                        out = [name, "[", pre] + flat_pairs + [c, post, params, "]"]

        return [self._str_categorical_begin()] + out + [self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, RecordType):
            if not _parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            ):
                return False

            if self._fields is None and other._fields is None:
                return self._contents == other._contents

            elif self._fields is not None and other._fields is not None:
                if set(self._fields) != set(other._fields):
                    return False
                for field in self._fields:
                    if self.content(field) != other.content(field):
                        return False

                return True

            else:
                return False

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
