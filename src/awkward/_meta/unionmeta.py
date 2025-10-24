# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections import Counter
from functools import cached_property

from awkward._meta.meta import Meta
from awkward._typing import Generic, JSONSerializable, TypeVar

T = TypeVar("T", bound=Meta)


class UnionMeta(Meta, Generic[T]):
    is_union = True

    _contents: list[T]

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

        for key in keys:
            out = self._contents[0].purelist_parameter(key)
            for content in self._contents[1:]:
                tmp = content.purelist_parameter(key)
                if out != tmp:
                    return None
            return out

        return None

    @cached_property
    def purelist_isregular(self) -> bool:  # type: ignore[override]
        for content in self._contents:
            if not content.purelist_isregular:
                return False
        return True

    @cached_property
    def purelist_depth(self) -> int:  # type: ignore[override]
        out = None
        for content in self._contents:
            if out is None:
                out = content.purelist_depth
            elif out != content.purelist_depth:
                return -1
        assert out is not None
        return out

    @property
    def is_identity_like(self) -> bool:
        return False

    @cached_property
    def minmax_depth(self) -> tuple[int, int]:  # type: ignore[override]
        if len(self._contents) == 0:
            return (0, 0)
        mindepth, maxdepth = self._contents[0].minmax_depth
        for content in self._contents[1:]:
            mindepth_, maxdepth_ = content.minmax_depth
            if mindepth_ < mindepth:
                mindepth = mindepth_
            if maxdepth_ > maxdepth:
                maxdepth = maxdepth_
        return (mindepth, maxdepth)

    @cached_property
    def branch_depth(self) -> tuple[bool, int]:  # type: ignore[override]
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

    @cached_property
    def fields(self) -> list[str]:  # type: ignore[override]
        field_counts = Counter([f for c in self._contents for f in c.fields])
        return [f for f, n in field_counts.items() if n == len(self._contents)]

    @property
    def is_tuple(self) -> bool:
        return all(x.is_tuple for x in self._contents) and (len(self._contents) > 0)

    @property
    def dimension_optiontype(self) -> bool:
        for content in self._contents:
            if content.dimension_optiontype:
                return True
        return False

    def content(self, index: int) -> T:
        return self._contents[index]

    @cached_property
    def contents(self) -> list[T]:
        return self._contents
