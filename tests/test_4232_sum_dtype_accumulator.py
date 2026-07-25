# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# The dtype= sum-accumulator feature was reverted (see PR discussion): the
# reducer already promotes int8/16/32 to int64, and float64 accumulation would
# slightly degrade the common non-overflowing int64 mean. This file is left
# empty and should be removed with `git rm`.

from __future__ import annotations
