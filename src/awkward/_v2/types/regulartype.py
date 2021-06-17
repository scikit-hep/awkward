# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import
import json
from awkward._v2.types.type import Type


class RegularType(Type):
    def __init__(self, content, size, parameters=None, typestr="unknown"):
        if not isinstance(content, Type):
            raise TypeError(
                "{0} 'content' must be a Type subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(size, int) or size < 0:
            raise ValueError(
                "{0} 'size' must be of a positive integer, not {1}".format(
                    type(self).__name__, repr(size)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._content = content
        self._size = size
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    def __str__(self):
        if self._typestr == "override" and (
            self._parameters is None or "__categorical__" not in self._parameters.keys()
        ):
            return "override"
        elif self._parameters is None:
            return "{0} * {1}".format(repr(self._size), self._typestr)
        elif (
            "__categorical__" in self._parameters.keys()
            and self._parameters["__categorical__"] is True
        ):
            if len(self._parameters) == 1:
                if self._typestr == "override":
                    return "categorical[type={0}]".format(self._typestr)
                else:
                    return "categorical[type={0} * {1}]".format(
                        repr(self._size), self._typestr
                    )
            else:
                return "categorical[type=[{0} * {1}, parameters={2}]]".format(
                    repr(self._size),
                    self._typestr,
                    json.dumps(
                        {
                            k: self._parameters[k]
                            for k in set(list(self._parameters.keys()))
                            - {"__categorical__"}
                        }
                    ),
                )
        elif "__array__" in self._parameters.keys():
            return "{0}[{1}]".format(self._parameters["__array__"], repr(self._size))
        else:
            return "[{0} * {1}, parameters={2}]".format(
                repr(self._size), self._typestr, json.dumps(self._parameters)
            )

    def __repr__(self):
        if self._parameters is None and self._typestr == "unknown":
            return "RegularType({0}, {1})".format(repr(self._content), repr(self._size))
        elif self._typestr == "unknown":
            return "RegularType({0}, {1}, parameters={2})".format(
                repr(self._content), repr(self._size), json.dumps(self._parameters)
            )
        else:
            return 'RegularType({0}, {1}, parameters={2}, typestr="{3}")'.format(
                repr(self._content),
                repr(self._size),
                json.dumps(self._parameters),
                self._typestr,
            )
