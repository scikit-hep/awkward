# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward.contents.content import Content

np = NumpyMetadata.instance()


def mergeable(one: Content, two: Content, mergebool: bool = True) -> bool:
    return one._mergeable_next(two, mergebool=mergebool)
