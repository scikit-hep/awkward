# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
from awkward._v2.types.type import Type


class OptionType(Type):
    def __init__(self, content, parameters=None, typestr="unknown"):
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
        self._content = content
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    def __str__(self):
        if hasattr(self._content, "content"):
            out = str(self._content)
        else:
            out = self._content.typestr
        if self._typestr == "override" and (
            self._parameters is None or "__categorical__" not in self._parameters.keys()
        ):
            return "override"
        elif self._parameters is None and self._typestr == "unknown":
            if hasattr(self._content, "content"):
                return "option[{0}]".format(out)
            else:
                return out
        elif self._parameters is None:
            return "{0}".format(self._typestr)
        elif (
            "__categorical__" in self._parameters.keys()
            and self._parameters["__categorical__"] is True
        ):
            if len(self._parameters) == 1:
                if self._typestr == "override":
                    return "categorical[type=override]"
                else:
                    if hasattr(self._content, "content"):
                        return "categorical[type=option[" + out + "]]"
                    else:
                        return "categorical[type=" + out + "]"
            else:
                if self._typestr == "override":
                    return "categorical[type=override]"
                else:
                    return "categorical[type=option[{0}, parameters={1}]]".format(
                        out,
                        json.dumps(
                            {
                                k: self._parameters[k]
                                for k in set(list(self._parameters.keys()))
                                - {"__categorical__"}
                            }
                        ),
                    )
        else:
            return "option[{0}, parameters={1}]".format(
                out, json.dumps(self._parameters)
            )

    def __repr__(self):
        if self._parameters is None and self._typestr == "unknown":
            return "OptionType({0})".format(repr(self._content))
        elif self._typestr == "unknown":
            return "OptionType({0}, parameters={1})".format(
                repr(self._content), json.dumps(self._parameters)
            )
        else:
            return 'OptionType({0}, parameters={1}, typestr="{2}")'.format(
                repr(self._content), json.dumps(self._parameters), self._typestr
            )
