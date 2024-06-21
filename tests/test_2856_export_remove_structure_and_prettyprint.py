# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations


def test_prettyprint_rename():
    import awkward._prettyprint as deprecated_prettyprint
    import awkward.prettyprint as new_prettyprint

    assert new_prettyprint.Formatter is deprecated_prettyprint.Formatter
    assert new_prettyprint.FormatterOptions is deprecated_prettyprint.FormatterOptions
    assert new_prettyprint.FormatterType is deprecated_prettyprint.FormatterType
    assert new_prettyprint.PlaceholderValue is deprecated_prettyprint.PlaceholderValue
    assert new_prettyprint.alternate is deprecated_prettyprint.alternate
    assert new_prettyprint.custom_str is deprecated_prettyprint.custom_str
    assert new_prettyprint.get_at is deprecated_prettyprint.get_at
    assert new_prettyprint.get_field is deprecated_prettyprint.get_field
    assert new_prettyprint.half is deprecated_prettyprint.half
    assert new_prettyprint.is_identifier is deprecated_prettyprint.is_identifier
    assert new_prettyprint.valuestr is deprecated_prettyprint.valuestr
    assert new_prettyprint.valuestr_horiz is deprecated_prettyprint.valuestr_horiz


def test_remove_structure_rename():
    from awkward._do import remove_structure as deprecated_remove_structure
    from awkward.contents.remove_structure import (
        remove_structure as new_remove_structure,
    )

    assert new_remove_structure is deprecated_remove_structure
