# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward._errors import deprecate
from awkward._nplikes.shape import ShapeItem
from awkward._typing import Iterator, final
from awkward._util import UNSET
from awkward.forms.form import Form, JSONMapping


@final
class EmptyForm(Form):
    is_numpy = True
    is_unknown = True

    def __init__(self, *, parameters: JSONMapping | None = None, form_key=None):
        if not (parameters is None or len(parameters) == 0):
            deprecate(
                f"{type(self).__name__} cannot contain parameters", version="2.2.0"
            )
        self._init(parameters=parameters, form_key=form_key)

    def copy(
        self, *, parameters: JSONMapping | None = UNSET, form_key=UNSET
    ) -> EmptyForm:
        if not (parameters is UNSET or parameters is None or len(parameters) == 0):
            deprecate(
                f"{type(self).__name__} cannot contain parameters", version="2.2.0"
            )
        return EmptyForm(
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(cls, *, parameters=None, form_key=None) -> Form:
        if not (parameters is None or len(parameters) == 0):
            deprecate(f"{cls.__name__} cannot contain parameters", version="2.2.0")
        return cls(parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra({"class": "EmptyArray"}, verbose)

    @property
    def type(self):
        return ak.types.UnknownType(parameters=self._parameters)

    def __eq__(self, other) -> bool:
        return isinstance(other, EmptyForm) and self._form_key == other._form_key

    def to_NumpyForm(self, dtype):
        return ak.forms.numpyform.from_dtype(dtype, parameters=self._parameters)

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return None
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self) -> bool:
        return True

    @property
    def purelist_depth(self) -> int:
        return 1

    @property
    def is_identity_like(self) -> bool:
        return True

    @property
    def minmax_depth(self) -> tuple[int, int]:
        return (1, 1)

    @property
    def branch_depth(self) -> tuple[bool, int]:
        return (False, 1)

    @property
    def fields(self) -> list[str]:
        return []

    @property
    def is_tuple(self) -> bool:
        return False

    @property
    def dimension_optiontype(self) -> bool:
        return False

    def _columns(self, path, output, list_indicator):
        output.append(".".join(path))

    def _select_columns(self, index, specifier, matches, output):
        if any(match and index >= len(item) for item, match in zip(specifier, matches)):
            output.append(None)
        return self

    def _column_types(self) -> tuple[str, ...]:
        return ("empty",)

    def _length_one_buffer_lengths(self) -> Iterator[ShapeItem]:
        yield 0
