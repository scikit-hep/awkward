# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._typing import (
    TYPE_CHECKING,
    ClassVar,
    JSONMapping,
    JSONSerializable,
    Self,
    TypeGuard,
)
from awkward._util import UNSET, Sentinel

if TYPE_CHECKING:
    from awkward._meta.bitmaskedmeta import BitMaskedMeta
    from awkward._meta.bytemaskedmeta import ByteMaskedMeta
    from awkward._meta.indexedmeta import IndexedMeta
    from awkward._meta.indexedoptionmeta import IndexedOptionMeta
    from awkward._meta.listmeta import ListMeta
    from awkward._meta.listoffsetmeta import ListOffsetMeta
    from awkward._meta.numpymeta import NumpyMeta
    from awkward._meta.recordmeta import RecordMeta
    from awkward._meta.regularmeta import RegularMeta
    from awkward._meta.unionmeta import UnionMeta
    from awkward._meta.unmaskedmeta import UnmaskedMeta


def is_option(
    meta: Meta
) -> TypeGuard[IndexedOptionMeta | BitMaskedMeta | ByteMaskedMeta | UnmaskedMeta]:
    return meta.is_option


def is_list(meta: Meta) -> TypeGuard[RegularMeta | ListOffsetMeta | ListMeta]:
    return meta.is_list


def is_numpy(meta: Meta) -> TypeGuard[NumpyMeta]:
    return meta.is_numpy


def is_regular(meta: Meta) -> TypeGuard[RegularMeta]:
    return meta.is_regular


def is_union(meta: Meta) -> TypeGuard[UnionMeta]:
    return meta.is_union


def is_record(meta: Meta) -> TypeGuard[RecordMeta]:
    return meta.is_record


def is_indexed(meta: Meta) -> TypeGuard[IndexedOptionMeta | IndexedMeta]:
    return meta.is_indexed


# FIXME: narrow this to have `is_tuple` be a const True
def is_record_tuple(meta: Meta) -> TypeGuard[RecordMeta]:
    return meta.is_record and meta.is_tuple


# FIXME: narrow this to have `is_tuple` be a const False
def is_record_record(meta: Meta) -> TypeGuard[RecordMeta]:
    return meta.is_record and not meta.is_tuple


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

    def _mergeable_next(self, other: Meta, mergebool: bool) -> bool:
        raise NotImplementedError
