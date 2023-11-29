# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._do.meta import (
    is_indexed,
    is_option,
    is_record,
    is_record_record,
    is_record_tuple,
)
from awkward._meta.meta import Meta
from awkward._parameters import type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import JSONSerializable
from awkward.errors import FieldNotFoundError


class RecordMeta(Meta):
    is_record = True

    _contents: list[Meta]
    _fields: list[str] | None

    @property
    def is_tuple(self) -> bool:
        return self._fields is None

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]
        return None

    @property
    def purelist_isregular(self) -> bool:
        return True

    @property
    def purelist_depth(self) -> int:
        return 1

    @property
    def is_identity_like(self) -> bool:
        return False

    @property
    def minmax_depth(self) -> tuple[int, int]:
        if len(self._contents) == 0:
            return (1, 1)
        mins, maxs = [], []
        for content in self._contents:
            mindepth, maxdepth = content.minmax_depth
            mins.append(mindepth)
            maxs.append(maxdepth)
        return (min(mins), max(maxs))

    @property
    def branch_depth(self) -> tuple[bool, int]:
        if len(self._contents) == 0:
            return False, 1

        any_branch = False
        min_depth = None
        for content in self._contents:
            branch, depth = content.branch_depth
            if min_depth is None:
                min_depth = depth
            if branch or min_depth != depth:
                any_branch = True
            if min_depth > depth:
                min_depth = depth
        assert min_depth is not None
        return any_branch, min_depth

    @property
    def dimension_optiontype(self) -> bool:
        return False

    @property
    def is_leaf(self) -> bool:  # type: ignore[override]
        return len(self._contents) == 0

    @property
    def contents(self) -> list[Meta]:
        return self._contents

    @property
    def fields(self) -> list[str]:
        if self._fields is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._fields

    def index_to_field(self, index: int) -> str:
        if 0 <= index < len(self._contents):
            if self._fields is None:
                return str(index)
            else:
                return self._fields[index]
        else:
            raise IndexError(
                f"no index {index} in record with {len(self._contents)} fields"
            )

    def field_to_index(self, field: str) -> int:
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

    def has_field(self, field: str) -> bool:
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                return False
            else:
                return 0 <= i < len(self._contents)
        else:
            return field in self._fields

    def content(self, index_or_field: int | str) -> Meta:
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
        return self._contents[index]  # type: ignore[index]

    def _mergeable_next(self, other: Meta, mergebool: bool) -> bool:
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_union:
            return True
        # Check against option contents
        elif is_option(other) or is_indexed(other):
            return self._mergeable_next(other.content, mergebool)
        # Otherwise, do the parameters match? If not, we can't merge.
        elif not type_parameters_equal(self._parameters, other._parameters):
            return False
        elif is_record(other):
            if is_record_tuple(self) and is_record_tuple(other):
                if len(self.contents) == len(other.contents):
                    for self_cont, other_cont in zip(self.contents, other.contents):
                        if not self_cont._mergeable_next(other_cont, mergebool):
                            return False

                    return True
                else:
                    return False

            elif is_record_record(self) and is_record_record(other):
                if set(self._fields) != set(other._fields):  # type: ignore[arg-type]
                    return False

                for i, field in enumerate(self._fields):  # type: ignore[arg-type]
                    x = self._contents[i]
                    y = other.contents[other.field_to_index(field)]
                    if not x._mergeable_next(y, mergebool):
                        return False
                return True

            else:
                return False

        else:
            return False
