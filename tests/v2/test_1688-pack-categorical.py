# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


numpy = ak.nplike.Numpy.instance()


def test():
    this = ak._v2.to_categorical(["one", "two", "one", "three", "one", "four"])
    assert ak._v2.is_categorical(this)
    # Ensure packing by itself doesn't change the type
    this_packed = ak._v2.packed(this)
    assert this_packed.type == this.type
    # Ensure the categories match between the two
    assert ak._v2.all(ak._v2.categories(this_packed) == ak._v2.categories(this))

    # Ensure the inner types match (ignoring the length change)
    this_subset_packed = ak._v2.packed(this[:-1])
    assert ak._v2.is_categorical(this_subset_packed)
    assert this_subset_packed.type.content == this.type.content
    # Ensure the categories match between the two
    assert ak._v2.all(ak._v2.categories(this_subset_packed) == ak._v2.categories(this))
