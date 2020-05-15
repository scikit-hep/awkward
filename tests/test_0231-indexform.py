# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import awkward1

def test():
    for itype in ["i8", "u8", "i32", "u32", "i64"]:
        form = awkward1.forms.ListOffsetForm(itype, awkward1.forms.EmptyForm())
        assert form.offsets == itype
