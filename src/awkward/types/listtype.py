# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward.forms.form import _parameters_equal
from awkward.types.type import Type


class ListType(Type):
    def __init__(self, content, *, parameters=None, typestr=None):
        if not isinstance(content, Type):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'content' must be a Type subtype, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'parameters' must be of type dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'typestr' must be of type string or None, not {}".format(
                        type(self).__name__, repr(typestr)
                    )
                )
            )
        self._content = content
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    def _str(self, indent, compact):
        if self._typestr is not None:
            out = [self._typestr]

        elif self.parameter("__array__") == "string":
            out = ["string"]

        elif self.parameter("__array__") == "bytestring":
            out = ["bytes"]

        else:
            params = self._str_parameters()
            if params is None:
                out = ["var * "] + self._content._str(indent, compact)
            else:
                out = (
                    ["[var * "] + self._content._str(indent, compact) + [f", {params}]"]
                )

        return [self._str_categorical_begin()] + out + [self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._content)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, ListType):
            return (
                _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
                and self._content == other._content
            )
        else:
            return False
