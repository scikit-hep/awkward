# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy
import numbers

from awkward_cpp.lib import _ext

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def parameters(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Extracts parameters from the outermost array node of `array` (many types
    supported, including all Awkward Arrays and Records).

    Parameters are a dict from str to JSON-like objects, usually strings.
    Every #ak.contents.Content node has a different set of parameters. Some
    key names are special, such as `"__record__"` and `"__array__"` that name
    particular records and arrays as capable of supporting special behaviors.

    See #ak.Array and #ak.behavior for a more complete description of
    behaviors.
    """
    with ak._errors.OperationErrorContext(
        "ak.parameters",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    if isinstance(array, (ak.highlevel.Array, ak.highlevel.Record)):
        return _copy(array.layout.parameters)

    elif isinstance(
        array,
        (ak.contents.Content, ak.record.Record),
    ):
        return _copy(array.parameters)

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        form = ak.forms.from_json(array.layout.form())
        return form.parameters

    elif isinstance(array, _ext.ArrayBuilder):
        form = ak.forms.from_json(array.form())
        return form.parameters

    else:
        return {}


def _copy(what):
    if all(isinstance(x, (str, numbers.Real)) for x in what.values()):
        return what.copy()
    else:
        return copy.deepcopy(what)
