# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable

import awkward as ak
from awkward._meta.bitmaskedmeta import BitMaskedMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, DType, Iterator, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher, index_to_dtype

__all__ = ("BitMaskedForm",)

np = NumpyMetadata.instance()


@final
class BitMaskedForm(BitMaskedMeta[Form], Form):
    _content: Form

    def __init__(
        self,
        mask,
        content,
        valid_when,
        lsb_order,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(mask, str):
            raise TypeError(
                f"{type(self).__name__} 'mask' must be of type str, not {mask!r}"
            )
        if not isinstance(content, Form):
            raise TypeError(
                f"{type(self).__name__} all 'contents' must be Form subclasses, not {content!r}"
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                f"{type(self).__name__} 'valid_when' must be bool, not {valid_when!r}"
            )
        if not isinstance(lsb_order, bool):
            raise TypeError(
                f"{type(self).__name__} 'lsb_order' must be bool, not {lsb_order!r}"
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._init(parameters=parameters, form_key=form_key)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def lsb_order(self):
        return self._lsb_order

    def copy(
        self,
        mask=UNSET,
        content=UNSET,
        valid_when=UNSET,
        lsb_order=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return BitMaskedForm(
            self._mask if mask is UNSET else mask,
            self._content if content is UNSET else content,
            self._valid_when if valid_when is UNSET else valid_when,
            self._lsb_order if lsb_order is UNSET else lsb_order,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        mask,
        content,
        valid_when,
        lsb_order,
        *,
        parameters=None,
        form_key=None,
    ):
        if content.is_union:
            return content._union_of_optionarrays("i64", parameters)
        elif content.is_indexed or content.is_option:
            return ak.forms.IndexedOptionForm.simplified(
                "i64",
                content,
                parameters=parameters,
            )
        else:
            return cls(
                mask,
                content,
                valid_when,
                lsb_order,
                parameters=parameters,
                form_key=form_key,
            )

    def __repr__(self):
        args = [
            repr(self._mask),
            repr(self._content),
            repr(self._valid_when),
            repr(self._lsb_order),
            *self._repr_args(),
        ]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "BitMaskedArray",
                "mask": self._mask,
                "valid_when": self._valid_when,
                "lsb_order": self._lsb_order,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.OptionType(
            self._content.type, parameters=self._parameters
        ).simplify_option_union()

    def _columns(self, path, output, list_indicator):
        self._content._columns(path, output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self | None:
        next_content = self._content._prune_columns(is_inside_record_or_union)
        if next_content is None:
            return None
        else:
            return self.copy(content=next_content)

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Self:
        return self.copy(content=self._content._select_columns(match_specifier))

    def _column_types(self):
        return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L155-L163
            (
                has_identities,
                parameters,
                form_key,
                mask,
                content,
                valid_when,
                lsb_order,
            ) = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(
                mask,
                content,
                valid_when,
                lsb_order,
                parameters=parameters,
                form_key=form_key,
            )

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        yield (getkey(self, "mask"), index_to_dtype[self._mask])
        if recursive:
            yield from self._content._expected_from_buffers(getkey, recursive)

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters, form_key)
            and self._mask == other._mask
            and self._valid_when == other._valid_when
            and self._lsb_order == other._lsb_order
            and self._content._is_equal_to(other._content, all_parameters, form_key)
        )
