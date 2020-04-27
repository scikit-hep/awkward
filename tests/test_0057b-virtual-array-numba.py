# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")
awkward1_connect_numba_arrayview = pytest.importorskip("awkward1._connect._numba.arrayview")

def test_form_layouts():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert numba.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)
