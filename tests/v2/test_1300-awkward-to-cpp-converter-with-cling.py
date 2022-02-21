# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ROOT = pytest.importorskip("ROOT")

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402


def test():
    array = ak._v2.Array([5, 4, 3, 2, 1])
    generator = ak._v2._connect.cling.togenerator(array.layout.form)
    lookup = ak._v2._lookup.Lookup(array.layout)

    generator.generate(ROOT.gInterpreter.Declare)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_1(ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.entry()};
  std::cout << "HEY " << obj[0] << std::endl;
}}
"""
    )

    ROOT.roottest_1(len(array), lookup.arrayptrs)
