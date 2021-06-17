# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import
import json
from awkward._v2.types.type import Type


class UnknownType(Type):
    def __init__(self, parameters=None, typestr="unknown"):
        if parameters != None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr != None and not isinstance(typestr, str):
            raise TypeError(
                "{0} 'typestr' must be of type string, not {1}".format(
                    type(self).__name__, repr(typestr)
                )
            )
        self._parameters = parameters
        self._typestr = typestr

    @property
    def parameters(self):
        return self._parameters

    @property
    def typestr(self):
        return self._typestr

    def __str__(self):
        if self._parameters == None:
            return self._typestr
        elif (
            self._typestr == "override"
            and "__categorical__" not in self._parameters.keys()
        ):
            return "override"
        elif (
            "__categorical__" in self._parameters.keys()
            and self._parameters["__categorical__"] == True
        ):
            if len(self._parameters) == 1:
                return "categorical[type={0}]".format(self._typestr)
            else:
                return "categorical[type={0}[parameters={1}]]".format(
                    self._typestr,
                    json.dumps(
                        {
                            k: self._parameters[k]
                            for k in set(list(self._parameters.keys()))
                            - {"__categorical__"}
                        }
                    ),
                )
        else:
            return "{0}[parameters={1}]".format(
                self._typestr, json.dumps(self._parameters)
            )

    def __repr__(self):
        if self._parameters == None and self._typestr == "unknown":
            return "UnknownType()"
        else:
            return 'UnknownType(parameters={0}, typestr="{1}")'.format(
                json.dumps(self._parameters), self._typestr
            )
