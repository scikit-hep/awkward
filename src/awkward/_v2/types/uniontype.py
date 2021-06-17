# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import json
from awkward._v2.types.type import Type


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
                "{0} 'parameters' must be of type dict, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._contents = contents
        self._parameters = parameters
        self._typestr = typestr

    @property
    def contents(self):
        return self._contents

    def __str__(self):
        primitives = []
        for c in self._contents:
            if hasattr(c, "primitive"):
                primitives.append(c.primitive)
            else:
                primitives.append("unknown")
        primitives = (", ").join(primitives)
        if self._typestr == "override" and (
            self._parameters is None or "__categorical__" not in self._parameters.keys()
        ):
            return "override"
        elif self._parameters is None:
            return "union[{0}]".format(primitives)
        elif (
            "__categorical__" in self._parameters.keys()
            and self._parameters["__categorical__"] is True
        ):
            if len(self._parameters) == 1:
                if self._typestr == "override":
                    return "categorical[type={0}]".format(self._typestr)
                else:
                    return "categorical[type=union[{0}]]".format(primitives)
            else:
                if self._typestr == "override":
                    return "categorical[type={0}]".format(self._typestr)
                else:
                    return "categorical[type=union[{0}, parameters={1}]]".format(
                        primitives,
                        json.dumps(
                            {
                                k: self._parameters[k]
                                for k in set(list(self._parameters.keys()))
                                - {"__categorical__"}
                            }
                        ),
                    )
        else:
            return "union[{0}, parameters={1}]".format(
                primitives,
                json.dumps(self._parameters),
            )

    def __repr__(self):
        contents_list = (", ").join(repr(x) for x in self._contents)
        if self._parameters is None and self._typestr is None:
            return "UnionType([{0}])".format(contents_list)
        elif self._typestr is None:
            return "UnionType([{0}], parameters={1})".format(
                contents_list, json.dumps(self._parameters)
            )
        else:
            return 'UnionType([{0}], parameters={1}, typestr="{2}")'.format(
                contents_list, json.dumps(self._parameters), self._typestr
            )
