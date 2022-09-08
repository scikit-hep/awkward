# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


numpy = ak.nplike.Numpy.instance()


def test():
    this = ak._v2.to_categorical(["one", "two", "one", "three", "one", "four"])
    # Ensure packing by itself doesn't change the type
    assert ak._v2.packed(this).type == this.type
    # Ensure the categories match between the two
    assert ak._v2.all(ak._v2.categories(ak._v2.packed(this)) == ak._v2.categories(this))
    # Ensure the inner types match (ignoring the length change)
    assert ak._v2.packed(this[:-1]).type.content == this.type.content
