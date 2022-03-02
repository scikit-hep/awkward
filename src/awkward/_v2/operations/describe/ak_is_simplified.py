# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def is_simplified(array, exception=False):
    """
    Args:
        array (#ak.Array, #ak.layout.Content, #ak.ArrayBuilder, #ak.layout.ArrayBuilder,
               #ak.layout.LayoutBuilder32, #ak.layout.LayoutBuilder64):
            Array to check.
        exception (bool): If True, unsimplified arrays raise exceptions.

    Returns True if there the array is already simplified and False if not.

    Checks for redundant layouts in the structure of the array, such as indexed types that wrap
    another indexed or option type. Either an error is raised or the function returns a boolean.
    """
    layout = ak._v2.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    def visitor(layout, depth, **kwargs):
        if layout.is_IndexedType or layout.is_OptionType:
            simplified = layout.simplify_optiontype()
        elif layout.is_UnionType:
            simplified = layout.simplify_uniontype()
        else:
            return

        if simplified.form != layout.form:
            raise ValueError(
                f"Form changed from\n{layout.form}\nto\n{simplified.form}\nafter simplification."
            )

    try:
        layout.recursively_apply(visitor, numpy_to_regular=False, return_array=False)
    except ValueError:
        if exception:
            raise
        return False
    else:
        return True
