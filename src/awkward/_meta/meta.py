# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._typing import (
    ClassVar,
    JSONMapping,
    JSONSerializable,
    Self,
)
from awkward._util import UNSET, Sentinel


class Meta:
    is_numpy: ClassVar[bool] = False
    is_unknown: ClassVar[bool] = False
    is_list: ClassVar[bool] = False
    is_regular: ClassVar[bool] = False
    is_option: ClassVar[bool] = False
    is_indexed: ClassVar[bool] = False
    is_record: ClassVar[bool] = False
    is_union: ClassVar[bool] = False
    is_leaf: ClassVar[bool] = False

    _parameters: JSONMapping | None

    @property
    def parameters(self) -> JSONMapping:
        """
        Free-form parameters associated with every array node as a dict from parameter
        name to its JSON-like value. Some parameters are special and are used to assign
        behaviors to the data.

        Note that the dict returned by this property is a *view* of the array node's
        parameters. *Changing the dict will change the array!*

        See #ak.behavior.
        """
        if self._parameters is None:
            self._parameters = {}
        return self._parameters

    def parameter(self, key: str) -> JSONSerializable:
        """
        Returns a parameter's value or None.

        (No distinction is ever made between unset parameters and parameters set to None.)
        """
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    def purelist_parameter(self, key: str) -> JSONSerializable:
        """
        Return the value of the outermost parameter matching `key` in a sequence
        of nested lists, stopping at the first record or tuple layer.

        If a layer has #ak.types.UnionType, the value is only returned if all
        possibilities have the same value.
        """
        return self.purelist_parameters(key)

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        """
        Return the value of the outermost parameter matching one of `keys` in a sequence
        of nested lists, stopping at the first record or tuple layer.

        If a layer has #ak.types.UnionType, the value is only returned if all
        possibilities have the same value.
        """
        raise NotImplementedError

    @property
    def is_identity_like(self) -> bool:
        raise NotImplementedError

    @property
    def purelist_isregular(self) -> bool:
        """
        Returns True if all dimensions down to the first record or tuple layer have
        #ak.types.RegularType; False otherwise.
        """
        raise NotImplementedError

    @property
    def purelist_depth(self) -> int:
        """
        Number of dimensions of nested lists, not counting anything deeper than the
        first record or tuple layer, if any. The depth of a one-dimensional array is
        `1`.

        If the array contains #ak.types.UnionType data and its contents have
        equal depths, the return value is that depth. If they do not have equal
        depths, the return value is `-1`.
        """
        raise NotImplementedError

    @property
    def minmax_depth(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def branch_depth(self) -> tuple[bool, int]:
        raise NotImplementedError

    @property
    def fields(self) -> list[str]:
        raise NotImplementedError

    @property
    def is_tuple(self) -> bool:
        raise NotImplementedError

    @property
    def dimension_optiontype(self) -> bool:
        raise NotImplementedError

    def copy(self, *, parameters: JSONMapping | None | Sentinel = UNSET) -> Self:
        raise NotImplementedError
