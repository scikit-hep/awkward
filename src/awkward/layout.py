# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: change to pull in classes from src/awkward/{index.py,record.py,contents/*.py}
# and add index-specific Content types as subclasses of the Python ones that enforce
# the index type. (Index already has subclasses that enforce integer type.)
# Maybe that subclassing belongs in the src/awkward/contents/*.py modules, but
# anyway they need to be exposed here for backward compatibility.
# Ignore the identities, VirtualArray stuff, kernel_lib, _PersistentSharedPtr, Iterator.

from awkward._ext import ArrayBuilder
from awkward._ext import LayoutBuilder32
from awkward._ext import LayoutBuilder64

from awkward._ext import kernel_lib


__all__ = [
    "ArrayBuilder",
    "LayoutBuilder32",
    "LayoutBuilder64",
    "kernel_lib",
]


def __dir__():
    return __all__
