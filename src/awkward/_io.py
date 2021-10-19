# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: drop this file, but the from_json functionality must be available somehow.
# (uproot_issue_90 will not be ported; it will be replaced by AwkwardForth.)

from __future__ import absolute_import

from awkward._ext import fromjson
from awkward._ext import uproot_issue_90


__all__ = ["fromjson", "uproot_issue_90"]


def __dir__():
    return __all__
