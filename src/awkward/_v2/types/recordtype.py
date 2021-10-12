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
    def __init__(self, contents, keys, parameters=None, typestr=None):
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
        if keys is not None and not isinstance(keys, Iterable):
            raise TypeError(
                "{0} 'keys' must be iterable, not {1}".format(
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
        self._keys = keys
        self._parameters = parameters
        self._typestr = typestr

    @property
    def contents(self):
        return self._contents

    @property
    def keys(self):
        return self._keys

    @property
    def is_tuple(self):
        return self._keys is None

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
                    pairs = [k + ": " + v for k, v in zip(self._keys, children)]
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
                        keys = [json.dumps(x) for x in self._keys]
                        out = "struct[[{0}], [{1}], {2}]".format(
                            ", ".join(keys), ", ".join(children), params
                        )
                    else:
                        pairs = [k + ": " + v for k, v in zip(self._keys, children)]
                        out = "{0}[{1}, {2}]".format(name, ", ".join(pairs), params)

        return self._str_categorical_begin() + out + self._str_categorical_end()

    def __repr__(self):
        args = [repr(self._contents), repr(self._keys)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, RecordType):
            if self._typestr != other._typestr or not _parameters_equal(
                self._parameters, other._parameters
            ):
                return False

            if self._keys is None and other._keys is None:
                return self._contents == other._contents

            elif self._keys is not None and other._keys is not None:
                if set(self._keys) != set(other._keys):
                    return False
                for key in self._keys:
                    if self.content(key) != other.content(key):
                        return False
                else:
                    return True

            else:
                return False

        else:
            return False

    def index_to_key(self, index):
        return ak._v2.forms.recordform.RecordForm.index_to_key(self, index)

    def key_to_index(self, key):
        return ak._v2.forms.recordform.RecordForm.key_to_index(self, key)

    def haskey(self, key):
        return ak._v2.forms.recordform.RecordForm.haskey(self, key)

    def content(self, index_or_key):
        return ak._v2.forms.recordform.RecordForm.content(self, index_or_key)
