# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._typing import JSONSerializable


class EmptyMeta(Meta):
    is_unknown = True
    is_leaf = True

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        return None

    @property
    def purelist_isregular(self) -> bool:
        return True

    @property
    def purelist_depth(self) -> int:
        return 1

    @property
    def is_identity_like(self) -> bool:
        return True

    @property
    def minmax_depth(self) -> tuple[int, int]:
        return (1, 1)

    @property
    def branch_depth(self) -> tuple[bool, int]:
        return (False, 1)

    @property
    def fields(self) -> list[str]:
        return []

    @property
    def is_tuple(self) -> bool:
        return False

    @property
    def dimension_optiontype(self) -> bool:
        return False
