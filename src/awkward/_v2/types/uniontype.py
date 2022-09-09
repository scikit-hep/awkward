# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

import awkward as ak
from awkward._v2.types.type import Type
from awkward._v2.forms.form import _parameters_equal


class UnionType(Type):
    def __init__(self, contents, parameters=None, typestr=None):
        if not isinstance(contents, Iterable):
            raise ak._v2._util.error(
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
                raise ak._v2._util.error(
                    TypeError(
                        "{} all 'contents' must be Type subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'parameters' must be of type dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )
        if typestr is not None and not ak._util.isstr(typestr):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'typestr' must be of type string or None, not {}".format(
                        type(self).__name__, repr(typestr)
                    )
                )
            )
        self._contents = contents
        self._parameters = parameters
        self._typestr = typestr

    @property
    def contents(self):
        return self._contents

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

            flat_children = [y for x in children for y in x]
            params = self._str_parameters()

            if params is None:
                out = ["union[", pre] + flat_children + [post, "]"]
            else:
                out = ["union[", pre] + flat_children + [", ", post, params, "]"]

        return [self._str_categorical_begin()] + out + [self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._contents)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, UnionType):
            return (
                _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
                and self._contents == other._contents
            )
        else:
            return False
