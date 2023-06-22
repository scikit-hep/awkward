# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from inspect import signature

import awkward as ak
from awkward._errors import deprecate
from awkward._nplikes.shape import ShapeItem
from awkward._typing import Iterator, Self, final
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

    def to_NumpyForm(self, *args, **kwargs):
        def legacy_impl(dtype):
            deprecate(
                f"the `dtype` parameter in {type(self).__name__}.to_NumpyForm is deprecated, "
                f"in favour of a new `primitive` argument. Pass `primitive` by keyword to opt-in to the new behavior.",
                version="2.4.0",
            )
            return ak.forms.numpyform.from_dtype(dtype, parameters=self._parameters)

        def new_impl(*, primitive):
            return ak.forms.numpyform.NumpyForm(primitive, parameters=self._parameters)

        dispatch_table = [
            new_impl,
            legacy_impl,
        ]
        for func in dispatch_table:
            sig = signature(func)
            try:
                bound_arguments = sig.bind(*args, **kwargs)
            except TypeError:
                continue
            else:
                return func(*bound_arguments.args, **bound_arguments.kwargs)
        raise AssertionError(
            f"{type(self).__name__}.to_NumpyForm accepts either the new `primitive` argument as a keyword-only "
            f"argument, or the legacy `dtype` argument as positional or keyword"
        )

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

    def _select_columns(self, match_specifier):
        return self

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self:
        return self

    def _column_types(self) -> tuple[str, ...]:
        return ("empty",)

    def _length_one_buffer_lengths(self) -> Iterator[ShapeItem]:
        yield 0
