# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._parameters import type_parameters_equal
from awkward._typing import Generic, JSONSerializable, TypeVar

T = TypeVar("T", bound=Meta)


class ListMeta(Meta, Generic[T]):
    is_list = True

    _content: T

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

        return self._content.purelist_parameters(*keys)

    @property
    def purelist_isregular(self) -> bool:
        return False

    @property
    def purelist_depth(self) -> int:
        if self.parameter("__array__") in ("string", "bytestring"):
            return 1
        else:
            return self._content.purelist_depth + 1

    @property
    def is_identity_like(self) -> bool:
        return False

    @property
    def minmax_depth(self) -> tuple[int, int]:
        if self.parameter("__array__") in ("string", "bytestring"):
            return (1, 1)
        else:
            mindepth, maxdepth = self._content.minmax_depth
            return (mindepth + 1, maxdepth + 1)

    @property
    def branch_depth(self) -> tuple[bool, int]:
        if self.parameter("__array__") in ("string", "bytestring"):
            return False, 1
        else:
            branch, depth = self._content.branch_depth
            return branch, depth + 1

    @property
    def fields(self):
        return self._content.fields

    @property
    def is_tuple(self) -> bool:
        return self._content.is_tuple

    @property
    def dimension_optiontype(self) -> bool:
        return False

    @property
    def content(self) -> T:
        return self._content

    def _mergeable_next(self, other: T, mergebool: bool) -> bool:
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_union:
            return True
        # Check against option contents
        elif other.is_option or other.is_indexed:
            return self._mergeable_next(other.content, mergebool)
        # Otherwise, do the parameters match? If not, we can't merge.
        elif not type_parameters_equal(self._parameters, other._parameters):
            return False
        elif other.is_list:
            return self._content._mergeable_next(other.content, mergebool)
        elif other.is_numpy and len(other.inner_shape) > 0:
            return self._mergeable_next(
                other._to_regular_primitive(), mergebool
            )  # TODO
        else:
            return False
