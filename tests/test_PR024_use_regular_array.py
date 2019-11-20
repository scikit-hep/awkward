# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_nothing():
    pass

# Sequential:
#############
# TODO: replace Content::getitem's promotion to ListArray with a promotion to RegularArray.
# TODO: ListArray's and ListOffsetArray's non-advanced getitem array should now output a RegularArray.
# TODO: all getitem arrays should handle non-flat SliceArray by wrapping in RegularArrays.
# TODO: all of the above should happen in Numba, too.

# Independent:
##############
# TODO: check the FIXME in awkward_listarray_getitem_next_array_advanced.
# TODO: setid should not be allowed on data that can be reached by multiple paths (which will break the ListArray ids above, unfortunately).
