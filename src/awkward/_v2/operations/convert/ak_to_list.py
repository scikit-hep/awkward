# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable
from collections.abc import Mapping

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_list(array):
    """
    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into Python objects. If `array` is not recognized as an array, it
    is passed through as-is.

    Awkward Array types have the following Pythonic translations.

       * #ak.types.NumpyType: converted into bool, int, float, datetimes, etc.
         (Same as NumPy's `ndarray.tolist`.)
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into list.
       * #ak.types.RegularType: also converted into list. Python (and JSON)
         forms lose information about the regularity of list lengths.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__bytestring__"`: converted into bytes.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__string__"`: converted into str.
       * #ak.types.RecordArray without field names: converted into tuple.
       * #ak.types.RecordArray with field names: converted into dict.
       * #ak.types.UnionArray: Python data are naturally heterogeneous.

    See also #ak.from_iter and #ak.Array.tolist.
    """
    if isinstance(
        array,
        (
            ak._v2.highlevel.Array,
            ak._v2.highlevel.Record,
            ak._v2.highlevel.ArrayBuilder,
        ),
    ):
        return array.to_list()

    elif isinstance(array, (ak._v2.contents.Content, ak._v2.record.Record)):
        return array.to_list(None)

    elif isinstance(
        array,
        (ak.layout.ArrayBuilder, ak.layout.LayoutBuilder32, ak.layout.LayoutBuilder64),
    ):
        formstr, length, container = array.to_buffers()
        form = ak._v2.forms.from_json(formstr)
        layout = ak._v2.operations.convert.from_buffers(form, length, container)
        return layout.to_list(None)

    elif isinstance(
        array,
        (
            ak.highlevel.Array,
            ak.highlevel.Record,
            ak.highlevel.ArrayBuilder,
            ak.layout.Content,
            ak.layout.Record,
        ),
    ):
        raise TypeError("do not use ak._v2.operations.convert.to_list on v1 arrays")

    elif hasattr(array, "tolist"):
        return array.tolist()

    elif hasattr(array, "to_list"):
        return array.to_list()

    elif isinstance(array, Mapping):
        return {k: to_list(v) for k, v in array.items()}

    elif isinstance(array, Iterable):
        return [to_list(x) for x in array]

    else:
        return array
