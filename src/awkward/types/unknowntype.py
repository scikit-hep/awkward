# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping

from awkward._typing import Any, JSONMapping, final
from awkward._util import UNSET
from awkward.types.type import Type


@final
class UnknownType(Type):
    def copy(self, *, parameters=UNSET) -> UnknownType:
        if not (parameters is UNSET or parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        return UnknownType()

    def __init__(self, *, parameters: JSONMapping | None = None):
        if not (parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        self._parameters: JSONMapping | None = None

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
        params = self._str_parameters()
        if params is None:
            out = ["unknown"]
        else:
            out = ["unknown[", params, "]"]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        return isinstance(other, type(self))
