# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import json
from awkward._v2.types.type import Type


class RecordType(Type):
    def __init__(self, contents, recordlookup, parameters=None, typestr="unknown"):
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
        if recordlookup is not None and not isinstance(recordlookup, Iterable):
            raise TypeError(
                "{0} 'recordlookup' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        self._contents = contents
        self._recordlookup = recordlookup
        self._parameters = parameters
        self._typestr = typestr

    @property
    def contents(self):
        return self._contents

    @property
    def recordlookup(self):
        return self._recordlookup

    def __str__(self):
        primitives_list = []
        for c in self._contents:
            if hasattr(c, "primitive"):
                primitives_list.append(c.primitive)
            else:
                primitives_list.append("unknown")
        if self._recordlookup is None:
            primitives = "(" + (", ").join(primitives_list) + ")"
        else:
            out = []
            for i in range(len(self._recordlookup)):
                out.append(
                    '"' + str(self._recordlookup[i]) + '": ' + primitives_list[i]
                )
            primitives = "{" + ", ".join(out) + "}"

        if self._typestr == "override" and (
            self._parameters is None or "__categorical__" not in self._parameters.keys()
        ):
            return "override"

        elif self._recordlookup is None and self._parameters is None:
            return primitives

        elif self._recordlookup is not None and self._parameters is None:
            return primitives

        elif (
            "__categorical__" in self._parameters.keys()
            and self._parameters["__categorical__"] is True
        ):
            if len(self._parameters) == 1:
                if self._typestr == "override":
                    return "categorical[type={0}]".format(self._typestr)
                else:
                    return "categorical[type={0}]".format(primitives)
            else:
                if self._typestr == "override":
                    return "categorical[type={0}]".format(self._typestr)
                elif "__record__" in self._parameters.keys():
                    if len(self._parameters) == 2:
                        return "categorical[type={0}[{1}]]".format(
                            self._parameters["__record__"], primitives[1:-1]
                        )
                    else:
                        return "categorical[type={0}[{1}, parameters={2}]]".format(
                            self._parameters["__record__"],
                            primitives[1:-1],
                            json.dumps(
                                {
                                    k: self._parameters[k]
                                    for k in set(list(self._parameters.keys()))
                                    - {"__categorical__", "__record__"}
                                }
                            ),
                        )
                else:
                    if self._recordlookup is None:
                        return "categorical[type=tuple[[{0}], parameters={1}]]".format(
                            primitives[1:-1],
                            json.dumps(
                                {
                                    k: self._parameters[k]
                                    for k in set(list(self._parameters.keys()))
                                    - {"__categorical__"}
                                }
                            ),
                        )
                    else:
                        return "categorical[type=struct[{0}, [{1}], parameters={2}]]".format(
                            json.dumps(self._recordlookup),
                            ", ".join(primitives_list),
                            json.dumps(
                                {
                                    k: self._parameters[k]
                                    for k in set(list(self._parameters.keys()))
                                    - {"__categorical__"}
                                }
                            ),
                        )

        elif "__array__" in self._parameters.keys():
            return "{0}".format(self._parameters["__array__"])

        elif "__record__" in self._parameters.keys():
            if len(self._parameters) == 1:
                return "{0}[{1}]".format(
                    self._parameters["__record__"], primitives[1:-1]
                )
            else:
                return "{0}[{1}, parameters={2}]".format(
                    self._parameters["__record__"],
                    primitives[1:-1],
                    json.dumps(
                        {
                            k: self._parameters[k]
                            for k in set(list(self._parameters.keys())) - {"__record__"}
                        }
                    ),
                )
        else:
            if self._recordlookup is None:
                return "tuple[[{0}], parameters={1}]".format(
                    primitives[1:-1], json.dumps(self._parameters)
                )
            else:
                return "struct[{0}, [{1}], parameters={2}]".format(
                    json.dumps(self._recordlookup),
                    ", ".join(primitives_list),
                    json.dumps(self._parameters),
                )

    def __repr__(self):
        contents_list = (", ").join(repr(x) for x in self._contents)
        lookup = (
            "None" if self._recordlookup is None else json.dumps(self._recordlookup)
        )
        if self._parameters is None and self._typestr == "unknown":
            return "RecordType([{0}], {1})".format(contents_list, lookup)
        elif self._typestr == "unknown":
            return "RecordType([{0}], {1}, parameters={2})".format(
                contents_list, self._recordlookup, json.dumps(self._parameters)
            )
        else:

            return 'RecordType([{0}], {1}, parameters={2}, typestr="{3}")'.format(
                contents_list, lookup, json.dumps(self._parameters), self._typestr
            )
