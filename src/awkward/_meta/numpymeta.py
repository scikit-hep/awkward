# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._nplikes.shape import ShapeItem
from awkward._typing import JSONSerializable


class NumpyMeta(Meta):
    is_numpy = True
    is_leaf = True
    inner_shape: tuple[ShapeItem, ...]

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
        return len(self.inner_shape) + 1

    @property
    def is_identity_like(self) -> bool:
        return False

    @property
    def minmax_depth(self) -> tuple[int, int]:
        depth = len(self.inner_shape) + 1
        return (depth, depth)

    @property
    def branch_depth(self) -> tuple[bool, int]:
        return (False, len(self.inner_shape) + 1)

    @property
    def fields(self) -> list[str]:
        return []

    @property
    def is_tuple(self) -> bool:
        return False

    @property
    def dimension_optiontype(self) -> bool:
        return False
