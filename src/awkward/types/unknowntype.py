# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from awkward._typing import Self, final
from awkward._util import UNSET
from awkward.types.type import Type


@final
class UnknownType(Type):
    def copy(self, *, parameters=UNSET) -> Self:
        if not (parameters is UNSET or parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        return UnknownType()

    def __init__(self, *, parameters=None):
        if not (parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        self._parameters = None

    def _str(self, indent, compact, behavior):
        params = self._str_parameters()
        if params is None:
            out = ["unknown"]
        else:
            out = ["unknown[", params, "]"]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other, all_parameters: bool):
        return isinstance(other, type(self))
