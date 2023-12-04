# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awkward._meta.meta import Meta

np = NumpyMetadata.instance()


def mergeable(one: Meta, two: Meta, mergebool: bool = True) -> bool:
    return one._mergeable_next(two, mergebool=mergebool)
