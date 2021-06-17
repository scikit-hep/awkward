# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import
import json
import re
from awkward._v2.types.type import Type

accepted_types = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "float128",
    "complex64",
    "complex128",
    "complex256",
    "datetime64",
    "timedelta64",
]


class NumpyType(Type):
    def __init__(self, primitive, parameters=None, typestr="unknown"):
        if primitive not in accepted_types:
            raise TypeError(
                "{0} 'primitive' must be either: {1}, not {2}".format(
                    type(self).__name__, repr(accepted_types), repr(parameters)
                )
            )
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
        self._primitive = primitive
        self._parameters = parameters
        self._typestr = typestr

    @property
    def primitive(self):
        return self._primitive

    @property
    def parameters(self):
        return self._parameters

    @property
    def typestr(self):
        return self._typestr

    def __str__(self):
        if self._parameters == None and self._typestr == "unknown":
            return self._primitive
        elif self._typestr == "override" and (
            self._parameters == None or "__categorical__" not in self._parameters.keys()
        ):
            return "override"
        elif (
            "__categorical__" in self._parameters.keys()
            and self._parameters["__categorical__"] == True
        ):
            if len(self._parameters) == 1:
                if self._typestr == "override":
                    return "categorical[type={0}]".format(self._typestr)
                else:
                    return "categorical[type={0}]".format(self._primitive)
            else:
                return "categorical[type={0}[parameters={1}]]".format(
                    self._primitive,
                    json.dumps(
                        {
                            k: self._parameters[k]
                            for k in set(list(self._parameters.keys()))
                            - {"__categorical__"}
                        }
                    ),
                )
        elif "__unit__" in self._parameters.keys():
            if re.match(r"1\D", self._parameters["__unit__"]):
                um = self._parameters["__unit__"][1:]
            else:
                um = self._parameters["__unit__"]
            if len(self._parameters) == 1:
                return "{0}[unit={1}]".format(self._primitive, json.dumps(um))
            else:
                return "{0}[unit={1}, parameters={2}]".format(
                    self._primitive,
                    json.dumps(um),
                    json.dumps(
                        {
                            k: self._parameters[k]
                            for k in set(list(self._parameters.keys())) - {"__unit__"}
                        }
                    ),
                )
        elif "__array__" in self._parameters.keys():
            return self._parameters["__array__"]
        else:
            return "{0}[parameters={1}]".format(
                self._primitive, json.dumps(self._parameters)
            )

    def __repr__(self):
        if self._parameters == None and self._typestr == "unknown":
            return 'NumpyType("{0}")'.format(self._primitive)
        elif self._typestr == "unknown":
            return 'NumpyType("{0}", parameters={1})'.format(
                self._primitive, json.dumps(self._parameters)
            )
        else:
            return 'NumpyType("{0}", parameters={1}, typestr="{2}")'.format(
                self._primitive, json.dumps(self._parameters), self._typestr
            )
