# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable, Mapping

from awkward_cpp.lib import _ext

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_list(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

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
    with ak._errors.OperationErrorContext(
        "ak.to_list",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    if isinstance(
        array,
        (
            ak.highlevel.Array,
            ak.highlevel.Record,
            ak.highlevel.ArrayBuilder,
        ),
    ):
        return array.to_list()

    elif isinstance(array, (ak.contents.Content, ak.record.Record)):
        return array.to_list(None)

    elif isinstance(array, _ext.ArrayBuilder):
        formstr, length, container = array.to_buffers()
        form = ak.forms.from_json(formstr)
        layout = ak.operations.from_buffers(form, length, container)
        return layout.to_list(None)

    elif hasattr(array, "tolist"):
        return array.tolist()

    elif hasattr(array, "to_list"):
        return array.to_list()

    elif isinstance(array, Mapping):
        return {k: _impl(v) for k, v in array.items()}

    elif isinstance(array, Iterable):
        return [_impl(x) for x in array]

    else:
        return array
