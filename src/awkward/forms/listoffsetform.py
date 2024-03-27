# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable

import awkward as ak
from awkward._meta.listoffsetmeta import ListOffsetMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import (
    Any,
    DType,
    Iterator,
    JSONMapping,
    Self,
    final,
)
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher, index_to_dtype

__all__ = ("ListOffsetForm",)

np = NumpyMetadata.instance()


@final
class ListOffsetForm(ListOffsetMeta[Form], Form):
    _content: Form

    def __init__(
        self,
        offsets: str,
        content: Form,
        *,
        parameters: JSONMapping | None = None,
        form_key: str | None = None,
    ):
        if not isinstance(offsets, str):
            raise TypeError(
                f"{type(self).__name__} 'offsets' must be of type str, not {offsets!r}"
            )

        self._offsets = offsets
        self._content = content
        self._init(parameters=parameters, form_key=form_key)

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    def copy(
        self,
        offsets=UNSET,
        content=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return ListOffsetForm(
            self._offsets if offsets is UNSET else offsets,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(cls, offsets, content, *, parameters=None, form_key=None):
        return cls(offsets, content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._offsets), repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "ListOffsetArray",
                "offsets": self._offsets,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.ListType(self._content.type, parameters=self._parameters)

    def _columns(self, path, output, list_indicator):
        if (
            self.parameter("__array__") not in ("string", "bytestring")
            and list_indicator is not None
        ):
            path = (*path, list_indicator)
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
        if self.parameter("__array__") in ("string", "bytestring"):
            return ("string",)
        else:
            return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L419-L425
            has_identities, parameters, form_key, offsets, content = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(offsets, content, parameters=parameters, form_key=form_key)

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        yield (getkey(self, "offsets"), index_to_dtype[self._offsets])
        if recursive:
            yield from self._content._expected_from_buffers(getkey, recursive)

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters, form_key)
            and self._offsets == other._offsets
            and self._content._is_equal_to(other._content, all_parameters, form_key)
        )
