# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: no change; keep this file.

from awkward_cpp.lib._ext import ForthMachine32, ForthMachine64

__all__ = ["ForthMachine32", "ForthMachine64"]


def __dir__():
    return __all__
