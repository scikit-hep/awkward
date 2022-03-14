# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import awkward._v2._connect.rdataframe.from_rdataframe  # noqa: F401


def from_rdataframe(
    rdf,
    builder_name="awkward::ArrayBuilderShim",
    function_for_foreach="my_function",
    highlevel=True,
    behavior=None,
):
    """
    Args:
        rdf (`ROOT.RDataFrame`): ROOT RDataFrame to convert into an
            Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts ROOT Data Frame columns into an Awkward Array.

    See also #ak.to_rdataframe.
    """
    import ROOT

    compiler = ROOT.gInterpreter.Declare

    return ak._v2._connect.rdataframe.from_rdataframe.generate_ArrayBuilder(compiler)
