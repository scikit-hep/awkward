# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import permutations

import awkward as ak
from awkward._meta.unionmeta import UnionMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, DType, Iterator, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher, index_to_dtype

__all__ = ("UnionForm",)

np = NumpyMetadata.instance()


@final
class UnionForm(UnionMeta[Form], Form):
    def __init__(
        self,
        tags,
        index,
        contents,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(tags, str):
            raise TypeError(
                f"{type(self).__name__} 'tags' must be of type str, not {tags!r}"
            )
        if not isinstance(index, str):
            raise TypeError(
                f"{type(self).__name__} 'index' must be of type str, not {index!r}"
            )
        if not isinstance(contents, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'contents' must be iterable, not {contents!r}"
            )
        for content in contents:
            if not isinstance(content, Form):
                raise TypeError(
                    f"{type(self).__name__} all 'contents' must be Form subclasses, not {content!r}"
                )

        self._tags = tags
        self._index = index
        self._contents = list(contents)
        self._init(parameters=parameters, form_key=form_key)

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    def copy(
        self,
        tags=UNSET,
        index=UNSET,
        contents=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return UnionForm(
            self._tags if tags is UNSET else tags,
            self._index if index is UNSET else index,
            self._contents if contents is UNSET else contents,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        tags,
        index,
        contents,
        *,
        parameters=None,
        form_key=None,
    ):
        return ak.contents.UnionArray.simplified(
            ak.index._form_to_zero_length(tags),
            ak.index._form_to_zero_length(index),
            [x.length_zero_array() for x in contents],
            parameters=parameters,
        ).form

    def _union_of_optionarrays(self, index, parameters):
        return (
            self.length_zero_array()
            ._union_of_optionarrays(ak.index._form_to_zero_length(index), parameters)
            .form
        )

    def __repr__(self):
        args = [
            repr(self._tags),
            repr(self._index),
            repr(self._contents),
            *self._repr_args(),
        ]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "UnionArray",
                "tags": self._tags,
                "index": self._index,
                "contents": [
                    content._to_dict_part(verbose, toplevel=False)
                    for content in self._contents
                ],
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.UnionType(
            [x.type for x in self._contents],
            parameters=self._parameters,
        )

    def _columns(self, path, output, list_indicator):
        for content, field in zip(self._contents, self.fields):
            content._columns((*path, field), output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool) -> Form | None:
        contents = []
        for content in self._contents:
            next_content = content._prune_columns(True)
            if next_content is None:
                continue
            contents.append(next_content)

        if len(contents) == 0:
            if is_inside_record_or_union:
                return None
            else:
                # outermost unions should return an EmptyForm instead
                return ak.forms.EmptyForm(form_key=self._form_key)
        elif len(contents) == 1:
            return contents[0]
        else:
            return self.copy(contents=contents)

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Self:
        contents = [
            content._select_columns(match_specifier) for content in self._contents
        ]

        return self.copy(contents=contents)

    def _column_types(self):
        return sum((x._column_types() for x in self._contents), ())

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L744-L755
            has_identities, parameters, form_key, tags, index, contents = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(
                tags, index, contents, parameters=parameters, form_key=form_key
            )

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        yield (getkey(self, "tags"), index_to_dtype[self._tags])
        yield (getkey(self, "index"), index_to_dtype[self._index])
        if recursive:
            for content in self._contents:
                yield from content._expected_from_buffers(getkey, recursive)

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters, form_key)
            and self._tags == other.tags
            and self._index == other.index
            and len(self._contents) == len(other.contents)
            and any(
                all(
                    x._is_equal_to(y, all_parameters, form_key)
                    for x, y in zip(self._contents, c)
                )
                for c in permutations(other.contents)
            )
        )
