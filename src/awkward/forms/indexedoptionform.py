# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable

import awkward as ak
from awkward._meta.indexedoptionmeta import IndexedOptionMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import (
    parameters_union,
)
from awkward._typing import Any, DType, Iterator, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher, index_to_dtype

__all__ = ("IndexedOptionForm",)

np = NumpyMetadata.instance()


@final
class IndexedOptionForm(IndexedOptionMeta[Form], Form):
    _content: Form

    def __init__(
        self,
        index,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(index, str):
            raise TypeError(
                f"{type(self).__name__} 'index' must be of type str, not {index!r}"
            )
        if not isinstance(content, Form):
            raise TypeError(
                f"{type(self).__name__} all 'contents' must be Form subclasses, not {content!r}"
            )

        self._index = index
        self._content = content
        self._init(parameters=parameters, form_key=form_key)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    def copy(
        self,
        index=UNSET,
        content=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return IndexedOptionForm(
            self._index if index is UNSET else index,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        index,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        is_cat = parameters is not None and parameters.get("__array__") == "categorical"

        if content.is_union and not is_cat:
            return content._union_of_optionarrays(index, parameters)

        elif content.is_indexed or content.is_option:
            return ak.forms.IndexedOptionForm.simplified(
                "i64",
                content.content,
                parameters=parameters_union(content._parameters, parameters),
            )

        else:
            return cls(index, content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._index), repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "IndexedOptionArray",
                "index": self._index,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        if self.parameter("__array__") == "categorical":
            parameters = dict(self._parameters)
            del parameters["__array__"]
            parameters["__categorical__"] = True
        else:
            parameters = self._parameters

        return ak.types.OptionType(
            self._content.type,
            parameters=parameters,
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

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L324-L330
            has_identities, parameters, form_key, index, content = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(index, content, parameters=parameters, form_key=form_key)

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        yield (getkey(self, "index"), index_to_dtype[self._index])
        if recursive:
            yield from self._content._expected_from_buffers(getkey, recursive)

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters, form_key)
            and self._index == other._index
            and self._content._is_equal_to(other._content, all_parameters, form_key)
        )
