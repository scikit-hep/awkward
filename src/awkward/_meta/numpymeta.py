# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


from __future__ import annotations

from functools import cached_property

from awkward._meta.meta import Meta
from awkward._nplikes.shape import ShapeItem
from awkward._typing import TYPE_CHECKING, DType, JSONSerializable

if TYPE_CHECKING:
    from awkward.forms.numpyform import NumpyForm
    from awkward.forms.regularform import RegularForm


class NumpyMeta(Meta):
    is_numpy = True
    is_leaf = True
    inner_shape: tuple[ShapeItem, ...]

    @property
    def dtype(self) -> DType:
        raise NotImplementedError

    @property
    def _mergeable_ndim(self) -> int:
        return len(self.inner_shape) + 1

    def _mergeable_regular(self) -> RegularForm | NumpyForm:
        # Mergeability ignores regular-list sizes. Use unit dimensions so that
        # even an unknown virtual shape never needs to be resolved.
        from awkward.forms.numpyform import NumpyForm
        from awkward.types.numpytype import dtype_to_primitive

        return NumpyForm(
            dtype_to_primitive(self.dtype),
            (1,) * (self._mergeable_ndim - 1),
            parameters=self._parameters,
        ).to_RegularForm()

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

    @cached_property
    def minmax_depth(self) -> tuple[int, int]:  # type: ignore[override]
        depth = len(self.inner_shape) + 1
        return (depth, depth)

    @cached_property
    def branch_depth(self) -> tuple[bool, int]:  # type: ignore[override]
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
