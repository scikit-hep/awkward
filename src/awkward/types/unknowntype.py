# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward.forms.form import _parameters_equal
from awkward.types.type import Type


class UnknownType(Type):
    def __init__(self, *, parameters=None, typestr=None):
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
        self._parameters = parameters
        self._typestr = typestr

    def _str(self, indent, compact):
        if self._typestr is not None:
            out = [self._typestr]

        else:
            params = self._str_parameters()
            if params is None:
                out = ["unknown"]
            else:
                out = ["unknown[", params, "]"]

        return [self._str_categorical_begin()] + out + [self._str_categorical_end()]

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, UnknownType):
            return _parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            )
        else:
            return False
