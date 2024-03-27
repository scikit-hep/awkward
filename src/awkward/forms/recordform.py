# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import awkward as ak
from awkward._meta.recordmeta import RecordMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, DType, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher

__all__ = ("RecordForm",)

np = NumpyMetadata.instance()


@final
class RecordForm(RecordMeta[Form], Form):
    def __init__(
        self,
        contents,
        fields,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(contents, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'contents' must be iterable, not {contents!r}"
            )
        for content in contents:
            if not isinstance(content, Form):
                raise TypeError(
                    f"{type(self).__name__} all 'contents' must be Form subclasses, not {content!r}"
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'fields' must be iterable, not {contents!r}"
            )

        self._fields = None if fields is None else list(fields)
        self._contents = list(contents)
        if fields is not None:
            assert len(self._fields) == len(self._contents)
        self._init(parameters=parameters, form_key=form_key)

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self) -> list[str]:
        if self._fields is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._fields

    def copy(
        self,
        contents=UNSET,
        fields=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return RecordForm(
            self._contents if contents is UNSET else contents,
            self._fields if fields is UNSET else fields,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        contents,
        fields,
        *,
        parameters=None,
        form_key=None,
    ):
        return cls(contents, fields, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        out = {"class": "RecordArray"}

        contents_tolist = [
            content._to_dict_part(verbose, toplevel=False) for content in self._contents
        ]
        if self._fields is not None:
            out["fields"] = list(self._fields)
        else:
            out["fields"] = None

        out["contents"] = contents_tolist
        return self._to_dict_extra(out, verbose)

    @property
    def type(self):
        return ak.types.RecordType(
            [x.type for x in self._contents],
            self._fields,
            parameters=self._parameters,
        )

    def _columns(self, path, output, list_indicator):
        for content, field in zip(self._contents, self.fields):
            content._columns((*path, field), output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self | None:
        contents = []
        fields = []
        for content, field in zip(self._contents, self.fields):
            next_content = content._prune_columns(True)
            if next_content is None:
                continue
            contents.append(next_content)
            fields.append(field)

        # If all subtrees are pruned (or we have no subtrees!), then we should prune this form too
        if not fields and is_inside_record_or_union:
            return None
        else:
            return self.copy(contents=contents, fields=fields)

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Self:
        contents = []
        fields = []
        for content, field in zip(self._contents, self.fields):
            # Try and match this field, allowing derived matcher to match any field if empty
            next_match_specifier = match_specifier(field, next_match_if_empty=True)
            if next_match_specifier is None:
                continue

            # NOTE: we could optimise this by avoiding column selection if we know the entire subtree will match
            #       this is given by `next_match_specifier.is_empty`, *but* we also want to perform column pruning
            #       meaning this optimisation is less useful, we we'd require a special prune-only pass
            next_content = content._select_columns(next_match_specifier)
            contents.append(next_content)
            fields.append(field)

        return self.copy(contents=contents, fields=fields)

    def _column_types(self):
        return sum((x._column_types() for x in self._contents), ())

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L624-L643
            has_identities, parameters, form_key, recordlookup, contents = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(
                contents, recordlookup, parameters=parameters, form_key=form_key
            )

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        if recursive:
            for content in self._contents:
                yield from content._expected_from_buffers(getkey, recursive)

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        computed_fields_set = set(self.fields)

        return (
            self._is_equal_to_generic(other, all_parameters, form_key)
            and self.is_tuple == other.is_tuple
            and len(self._contents) == len(other._contents)
            and all(f in computed_fields_set for f in other.fields)
            and all(
                content._is_equal_to(other.content(field), all_parameters, form_key)
                for field, content in zip(self.fields, self._contents)
            )
        )
