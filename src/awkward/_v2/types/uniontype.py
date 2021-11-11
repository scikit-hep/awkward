# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.types.type import Type
from awkward._v2.forms.form import _parameters_equal


class UnionType(Type):
    def __init__(self, contents, parameters=None, typestr=None):
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
        self._parameters = parameters
        self._typestr = typestr

    @property
    def contents(self):
        return self._contents

    def __str__(self):
        if self._typestr is not None:
            out = self._typestr

        else:
            children = [str(x) for x in self._contents]
            params = self._str_parameters()

            if params is None:
                out = "union[{0}]".format(", ".join(children))
            else:
                out = "union[{0}, {1}]".format(", ".join(children), params)

        return self._str_categorical_begin() + out + self._str_categorical_end()

    def __repr__(self):
        args = [repr(self._contents)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, UnionType):
            return (
                self._typestr == other._typestr
                and _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
                and self._contents == other._contents
            )
        else:
            return False
