# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

def test():
    for itype in ["i8", "u8", "i32", "u32", "i64"]:
        form = ak.forms.ListOffsetForm(itype, ak.forms.EmptyForm())
        assert form.offsets == itype
