# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable

import awkward as ak
from awkward._meta.emptymeta import EmptyMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._typing import DType, Iterator, Self, final
from awkward._util import UNSET, Sentinel
from awkward.forms.form import Any, Form, JSONMapping, _SpecifierMatcher

__all__ = ("EmptyForm",)

np = NumpyMetadata.instance()


@final
class EmptyForm(EmptyMeta, Form):
    def __init__(
        self, *, parameters: JSONMapping | None = None, form_key: str | None = None
    ):
        if not (parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        self._init(parameters=parameters, form_key=form_key)

    def copy(
        self,
        *,
        parameters: JSONMapping | Sentinel | None = UNSET,
        form_key: str | Sentinel | None = UNSET,
    ) -> EmptyForm:
        if not (parameters is UNSET or parameters is None or len(parameters) == 0):  # type: ignore[arg-type]
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        return EmptyForm(
            form_key=self._form_key if form_key is UNSET else form_key,  # type: ignore[arg-type]
        )

    @classmethod
    def simplified(
        cls, *, parameters: JSONMapping | None = None, form_key: str | None = None
    ) -> Form:
        if not (parameters is None or len(parameters) == 0):
            raise TypeError(f"{cls.__name__} cannot contain parameters")
        return cls(parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra({"class": "EmptyArray"}, verbose)

    @property
    def type(self):
        return ak.types.UnknownType()

    def to_NumpyForm(self, primitive):
        return ak.forms.numpyform.NumpyForm(primitive)

    def _columns(self, path, output, list_indicator):
        output.append(".".join(path))

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Self:
        return self

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self:
        return self

    def _column_types(self) -> tuple[str, ...]:
        return ("empty",)

    def _length_one_buffer_lengths(self) -> Iterator[ShapeItem]:
        yield 0

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L240-L244
            has_identities, parameters, form_key = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(form_key=form_key)

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        yield from ()

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        return isinstance(other, type(self)) and not (
            form_key and self._form_key != other._form_key
        )
