# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


from __future__ import annotations

from functools import cached_property

from awkward._meta.meta import Meta
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._parameters import type_parameters_equal
from awkward._typing import TYPE_CHECKING, DType, JSONSerializable, Literal

if TYPE_CHECKING:
    from awkward._meta.mergeable import _Wrapper
    from awkward.forms.numpyform import NumpyForm
    from awkward.forms.regularform import RegularForm


np = NumpyMetadata.instance()
numpy = Numpy.instance()


class NumpyMeta(Meta):
    is_numpy = True
    is_leaf = True
    inner_shape: tuple[ShapeItem, ...]

    @property
    def _mergeable_dtype(self) -> DType:
        """Primitive dtype for internal compatibility checks, without buffers."""
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
            dtype_to_primitive(self._mergeable_dtype),
            (1,) * (self._mergeable_ndim - 1),
            parameters=self._parameters,
        ).to_RegularForm()

    def _mergeable_next(
        self,
        other: Meta,
        mergebool: bool,
        mergecastable: Literal["same_kind", "equiv", "family"],
    ) -> bool:
        if other.is_identity_like or other.is_union:
            return True
        if other.is_indexed or other.is_option:
            if TYPE_CHECKING:
                assert isinstance(other, _Wrapper)
            return self._mergeable_next(other.content, mergebool, mergecastable)
        if not type_parameters_equal(self._parameters, other._parameters):
            return False
        if self._mergeable_ndim > 1:
            return self._mergeable_regular()._mergeable_next(
                other, mergebool, mergecastable
            )
        if not isinstance(other, NumpyMeta) or other._mergeable_ndim > 1:
            return False
        left = self._mergeable_dtype
        right = other._mergeable_dtype
        if left == right:
            return True
        if (np.issubdtype(left, np.bool_) and np.issubdtype(right, np.number)) or (
            np.issubdtype(left, np.number) and np.issubdtype(right, np.bool_)
        ):
            return mergebool
        if any(
            np.issubdtype(dtype, temporal)
            for dtype in (left, right)
            for temporal in (np.datetime64, np.timedelta64)
        ):
            return False
        if mergecastable == "family":
            return any(
                np.issubdtype(left, family) and np.issubdtype(right, family)
                for family in (np.integer, np.floating, np.complexfloating)
            )
        if mergecastable in ("same_kind", "equiv"):
            # Casting compares dtype descriptors, never device buffers. Use the
            # same NumPy rule for forms, layouts and mixed-backend metadata.
            return numpy.can_cast(left, right, casting=mergecastable) or numpy.can_cast(
                right, left, casting=mergecastable
            )
        raise TypeError(f"unrecognized mergecastable option: {mergecastable}")

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
