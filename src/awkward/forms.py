# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: replace with src/awkward/_v2/forms.

from __future__ import absolute_import

from awkward._ext import Form
from awkward._ext import BitMaskedForm
from awkward._ext import ByteMaskedForm
from awkward._ext import EmptyForm
from awkward._ext import IndexedForm
from awkward._ext import IndexedOptionForm
from awkward._ext import ListForm
from awkward._ext import ListOffsetForm
from awkward._ext import NumpyForm
from awkward._ext import RecordForm
from awkward._ext import RegularForm
from awkward._ext import UnionForm
from awkward._ext import UnmaskedForm
from awkward._ext import VirtualForm


__all__ = [
    "Form",
    "BitMaskedForm",
    "ByteMaskedForm",
    "EmptyForm",
    "IndexedForm",
    "IndexedOptionForm",
    "ListForm",
    "ListOffsetForm",
    "NumpyForm",
    "RecordForm",
    "RegularForm",
    "UnionForm",
    "UnmaskedForm",
    "VirtualForm",
]


def __dir__():
    return __all__
