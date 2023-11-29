# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._do.meta import is_indexed, is_numpy, is_option
from awkward._meta.meta import Meta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._parameters import type_parameters_equal
from awkward._typing import TYPE_CHECKING, DType, JSONSerializable

np = NumpyMetadata.instance()
if TYPE_CHECKING:
    from awkward._meta.regularmeta import RegularMeta


class NumpyMeta(Meta):
    is_numpy = True
    is_leaf = True
    inner_shape: tuple[ShapeItem, ...]

    @property
    def dtype(self) -> DType:
        raise NotImplementedError

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

    def _to_regular_primitive(self) -> RegularMeta | NumpyMeta:
        raise NotImplementedError

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
        # Simplify *this* branch to be 1D self
        elif len(self.inner_shape) > 0:
            return self._to_regular_primitive()._mergeable_next(
                other, mergebool
            )  # TODO

        elif is_numpy(other):
            if len(self.inner_shape) != len(other.inner_shape):
                return False

            # Obvious fast-path
            if self.dtype == other.dtype:  # TODO
                return True

            # Special-case booleans i.e. {bool, number}
            elif (
                np.issubdtype(self.dtype, np.bool_)
                and np.issubdtype(other.dtype, np.number)
                or np.issubdtype(self.dtype, np.number)
                and np.issubdtype(other.dtype, np.bool_)
            ):
                return mergebool

            # Currently we're less permissive than NumPy on merging datetimes / timedeltas
            elif (
                np.issubdtype(self.dtype, np.datetime64)
                or np.issubdtype(self.dtype, np.timedelta64)
                or np.issubdtype(other.dtype, np.datetime64)
                or np.issubdtype(other.dtype, np.timedelta64)
            ):
                return False

            # Default merging (can we cast one to the other)
            else:
                return np.can_cast(
                    self.dtype, other.dtype, casting="same_kind"
                ) or np.can_cast(other.dtype, self.dtype, casting="same_kind")

        else:
            return False
