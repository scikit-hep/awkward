# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: change to pull in classes from src/awkward/_v2/{index.py,record.py,contents/*.py}
# and add index-specific Content types as subclasses of the Python ones that enforce
# the index type. (Index already has subclasses that enforce integer type.)
# Maybe that subclassing belongs in the src/awkward/_v2/contents/*.py modules, but
# anyway they need to be exposed here for backward compatibility.
# Ignore the identities, VirtualArray stuff, kernel_lib, _PersistentSharedPtr, Iterator.

from __future__ import absolute_import

from awkward._ext import Index8
from awkward._ext import IndexU8
from awkward._ext import Index32
from awkward._ext import IndexU32
from awkward._ext import Index64

from awkward._ext import Identities32
from awkward._ext import Identities64

from awkward._ext import Iterator
from awkward._ext import ArrayBuilder
from awkward._ext import LayoutBuilder32
from awkward._ext import LayoutBuilder64
from awkward._ext import _PersistentSharedPtr

from awkward._ext import Content

from awkward._ext import EmptyArray

from awkward._ext import IndexedArray32
from awkward._ext import IndexedArrayU32
from awkward._ext import IndexedArray64
from awkward._ext import IndexedOptionArray32
from awkward._ext import IndexedOptionArray64

from awkward._ext import ByteMaskedArray
from awkward._ext import BitMaskedArray
from awkward._ext import UnmaskedArray

from awkward._ext import ListArray32
from awkward._ext import ListArrayU32
from awkward._ext import ListArray64

from awkward._ext import ListOffsetArray32
from awkward._ext import ListOffsetArrayU32
from awkward._ext import ListOffsetArray64

from awkward._ext import NumpyArray

from awkward._ext import Record
from awkward._ext import RecordArray

from awkward._ext import RegularArray

from awkward._ext import UnionArray8_32
from awkward._ext import UnionArray8_U32
from awkward._ext import UnionArray8_64

from awkward._ext import VirtualArray
from awkward._ext import ArrayGenerator
from awkward._ext import SliceGenerator
from awkward._ext import ArrayCache

from awkward._ext import kernel_lib


__all__ = [
    "Index8",
    "IndexU8",
    "Index32",
    "IndexU32",
    "Index64",
    "Identities32",
    "Identities64",
    "Iterator",
    "ArrayBuilder",
    "LayoutBuilder32",
    "LayoutBuilder64",
    "_PersistentSharedPtr",
    "Content",
    "EmptyArray",
    "IndexedArray32",
    "IndexedArrayU32",
    "IndexedArray64",
    "IndexedOptionArray32",
    "IndexedOptionArray64",
    "ByteMaskedArray",
    "BitMaskedArray",
    "UnmaskedArray",
    "ListArray32",
    "ListArrayU32",
    "ListArray64",
    "ListOffsetArray32",
    "ListOffsetArrayU32",
    "ListOffsetArray64",
    "NumpyArray",
    "Record",
    "RecordArray",
    "RegularArray",
    "UnionArray8_32",
    "UnionArray8_U32",
    "UnionArray8_64",
    "VirtualArray",
    "ArrayGenerator",
    "SliceGenerator",
    "ArrayCache",
    "kernel_lib",
]


def __dir__():
    return __all__
