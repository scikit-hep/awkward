# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import json

import awkward as ak
from awkward._v2.types.type import Type
from awkward._v2.forms.form import _parameters_equal


class RecordType(Type):
    def __init__(self, contents, fields, parameters=None, typestr=None):
        if not isinstance(contents, Iterable):
            raise TypeError(
                "{0} 'contents' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)
        for content in contents:
            if not isinstance(content, Type):
                raise TypeError(
                    "{0} all 'contents' must be Type subclasses, not {1}".format(
                        type(self).__name__, repr(content)
                    )
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise TypeError(
                "{0} 'fields' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr is not None and not ak._util.isstr(typestr):
            raise TypeError(
                "{0} 'typestr' must be of type string or None, not {1}".format(
                    type(self).__name__, repr(typestr)
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

    def __str__(self):
        if self._typestr is not None:
            out = self._typestr

        else:
            children = [str(x) for x in self._contents]
            params = self._str_parameters()
            name = self.parameter("__record__")

            if params is None:
                if self.is_tuple:
                    if name is None:
                        out = "(" + ", ".join(children) + ")"
                    else:
                        out = name + "[" + ", ".join(children) + "]"
                else:
                    pairs = [k + ": " + v for k, v in zip(self._fields, children)]
                    if name is None:
                        out = "{" + ", ".join(pairs) + "}"
                    else:
                        out = name + "[" + ", ".join(pairs) + "]"

            else:
                if self.is_tuple:
                    if name is None:
                        out = "tuple[[{0}], {1}]".format(", ".join(children), params)
                    else:
                        out = "{0}[{1}, {2}]".format(name, ", ".join(children), params)
                else:
                    if name is None:
                        fields = [json.dumps(x) for x in self._fields]
                        out = "struct[[{0}], [{1}], {2}]".format(
                            ", ".join(fields), ", ".join(children), params
                        )
                    else:
                        pairs = [k + ": " + v for k, v in zip(self._fields, children)]
                        out = "{0}[{1}, {2}]".format(name, ", ".join(pairs), params)

        return self._str_categorical_begin() + out + self._str_categorical_end()

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, RecordType):
            if self._typestr != other._typestr or not _parameters_equal(
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
                else:
                    return True

            else:
                return False

        else:
            return False

    def index_to_field(self, index):
        return ak._v2.forms.recordform.RecordForm.index_to_field(self, index)

    def field_to_index(self, field):
        return ak._v2.forms.recordform.RecordForm.field_to_index(self, field)

    def has_field(self, field):
        return ak._v2.forms.recordform.RecordForm.has_field(self, field)

    def content(self, index_or_field):
        return ak._v2.forms.recordform.RecordForm.content(self, index_or_field)
