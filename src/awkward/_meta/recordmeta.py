# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._regularize import is_integer
from awkward._typing import Generic, JSONSerializable, TypeVar
from awkward.errors import FieldNotFoundError

T = TypeVar("T", bound=Meta)


class RecordMeta(Meta, Generic[T]):
    is_record = True

    _contents: list[T]
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
    def contents(self) -> list[T]:
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

    def content(self, index_or_field: int | str) -> T:
        if is_integer(index_or_field):
            index = index_or_field
        elif isinstance(index_or_field, str):
            index = self.field_to_index(index_or_field)
        else:
            raise TypeError(
                f"index_or_field must be an integer (index) or string (field), not {index_or_field!r}"
            )
        return self._contents[index]  # type: ignore[index]
