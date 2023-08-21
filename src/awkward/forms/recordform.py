# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("RecordForm",)
from collections.abc import Callable, Iterable, Iterator

import awkward as ak
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._parameters import type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import JSONSerializable, final
from awkward._util import UNSET
from awkward.errors import FieldNotFoundError
from awkward.forms.form import Form

np = NumpyMetadata.instance()


@final
class RecordForm(Form):
    is_record = True

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
                "{} 'contents' must be iterable, not {}".format(
                    type(self).__name__, repr(contents)
                )
            )
        for content in contents:
            if not isinstance(content, Form):
                raise TypeError(
                    "{} all 'contents' must be Form subclasses, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'fields' must be iterable, not {contents!r}"
            )

        self._fields = None if fields is None else list(fields)
        self._contents = list(contents)
        self._init(parameters=parameters, form_key=form_key)

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self):
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

    @property
    def is_tuple(self):
        return self._fields is None

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def index_to_field(self, index):
        if 0 <= index < len(self._contents):
            if self._fields is None:
                return str(index)
            else:
                return self._fields[index]
        else:
            raise IndexError(
                f"no index {index} in record with {len(self._contents)} fields"
            )

    def field_to_index(self, field):
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._fields.index(field)
            except ValueError:
                pass
            else:
                return i
        raise FieldNotFoundError(
            f"no field {field!r} in record with {len(self._contents)} fields"
        )

    def has_field(self, field):
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                return False
            else:
                return 0 <= i < len(self._contents)
        else:
            return field in self._fields

    def content(self, index_or_field):
        if is_integer(index_or_field):
            index = index_or_field
        elif isinstance(index_or_field, str):
            index = self.field_to_index(index_or_field)
        else:
            raise TypeError(
                "index_or_field must be an integer (index) or string (field), not {}".format(
                    repr(index_or_field)
                )
            )
        return self._contents[index]

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

    def __eq__(self, other):
        if isinstance(other, RecordForm):
            if (
                self._form_key == other._form_key
                and self.is_tuple == other.is_tuple
                and len(self._contents) == len(other._contents)
                and type_parameters_equal(self._parameters, other._parameters)
            ):
                if self.is_tuple:
                    return self._contents == other._contents
                else:
                    return dict(zip(self._fields, self._contents)) == dict(
                        zip(other._fields, other._contents)
                    )
            else:
                return False
        else:
            return False

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def is_identity_like(self):
        return False

    @property
    def minmax_depth(self):
        if len(self._contents) == 0:
            return (1, 1)
        mins, maxs = [], []
        for content in self._contents:
            mindepth, maxdepth = content.minmax_depth
            mins.append(mindepth)
            maxs.append(maxdepth)
        return (min(mins), max(maxs))

    @property
    def branch_depth(self):
        if len(self._contents) == 0:
            return (False, 1)
        anybranch = False
        mindepth = None
        for content in self._contents:
            branch, depth = content.branch_depth
            if mindepth is None:
                mindepth = depth
            if branch or mindepth != depth:
                anybranch = True
            if mindepth > depth:
                mindepth = depth
        return (anybranch, mindepth)

    @property
    def dimension_optiontype(self):
        return False

    def _columns(self, path, output, list_indicator):
        for content, field in zip(self._contents, self.fields):
            content._columns((*path, field), output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool):
        contents = []
        fields = []
        for content, field in zip(self._contents, self.fields):
            next_content = content._prune_columns(True)
            if next_content is None:
                continue
            contents.append(next_content)
            fields.append(field)

        if fields or not is_inside_record_or_union:
            return RecordForm(
                contents,
                fields,
                parameters=self._parameters,
                form_key=self._form_key,
            )
        # If all subtrees are pruned (or we have no subtrees!), then we should prune this form too
        else:
            return None

    def _select_columns(self, match_specifier):
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

        return RecordForm(
            contents,
            fields,
            parameters=self._parameters,
            form_key=self._form_key,
        )

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
        self, getkey: Callable[[Form, str], str]
    ) -> Iterator[tuple[str, np.dtype]]:
        for content in self._contents:
            yield from content._expected_from_buffers(getkey)
