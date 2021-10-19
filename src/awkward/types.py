# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: change to pull in classes from src/awkward/_v2/types/*.py.

from __future__ import absolute_import

# Typeparser
from awkward._typeparser.parser import from_datashape

# Types
from awkward._ext import Type
from awkward._ext import ArrayType
from awkward._ext import PrimitiveType
from awkward._ext import RegularType
from awkward._ext import UnknownType
from awkward._ext import ListType
from awkward._ext import OptionType
from awkward._ext import UnionType
from awkward._ext import RecordType


__all__ = [
    "from_datashape",
    "Type",
    "ArrayType",
    "PrimitiveType",
    "RegularType",
    "UnknownType",
    "ListType",
    "OptionType",
    "UnionType",
    "RecordType",
]


def __dir__():
    return __all__
