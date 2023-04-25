# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from awkward._errors import deprecate
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._typing import final
from awkward.types.type import Type


@final
class UnknownType(Type):
    def __init__(self, *, parameters=None, typestr=None):
        if parameters is not None:
            deprecate(
                f"{type(self).__name__} cannot contain parameters", version="2.2.0"
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{} 'parameters' must be of type dict or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise TypeError(
                "{} 'typestr' must be of type string or None, not {}".format(
                    type(self).__name__, repr(typestr)
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

        return [self._str_categorical_begin(), *out] + [self._str_categorical_end()]

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other, all_parameters: bool):
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return isinstance(other, type(self)) and compare_parameters(
            self._parameters, other._parameters
        )
