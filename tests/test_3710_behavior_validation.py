# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_only_floats_validation():
    class OnlyFloatsError(Exception): ...

    class OnlyFloats(ak.Array):
        def __awkward_validation__(self) -> None:
            def _content_only_floats(layout, **kwargs) -> None:
                del kwargs
                if layout.is_numpy:
                    if not np.issubdtype(layout.dtype, np.floating):
                        raise OnlyFloatsError(
                            "OnlyFloats arrays can only contain floating-point numbers."
                        )

            _ = ak.transform(_content_only_floats, self, return_value="none")

    behavior = {
        "OnlyFloats": OnlyFloats,
        ("*", "OnlyFloats"): OnlyFloats,
    }

    # this should pass
    _ = ak.with_parameter([1.0], "__list__", "OnlyFloats", behavior=behavior)

    # this should fail
    with pytest.raises(OnlyFloatsError):
        _ = ak.with_parameter([1], "__list__", "OnlyFloats", behavior=behavior)


def test_field_validation():
    class OnlyAllowedFieldsError(Exception): ...

    class OnlyAllowedFields(ak.Array):
        _allowed_fields = frozenset({"pt", "eta", "phi", "mass"})

        def __awkward_validation__(self) -> None:
            fields = set(self.fields)
            if self._allowed_fields != fields:
                raise OnlyAllowedFieldsError(
                    f"OnlyAllowedFields arrays can only have fields: {self._allowed_fields}, "
                    f"but found fields: {fields}"
                )

    behavior = {
        "OnlyAllowedFields": OnlyAllowedFields,
        ("*", "OnlyAllowedFields"): OnlyAllowedFields,
    }

    # this should pass
    _ = ak.with_parameter(
        [{"pt": 1.0, "eta": 1.0, "phi": 1.0, "mass": 1.0}],
        "__list__",
        "OnlyAllowedFields",
        behavior=behavior,
    )

    # this should fail
    with pytest.raises(
        OnlyAllowedFieldsError, match="OnlyAllowedFields arrays can only have fields"
    ):
        _ = ak.with_parameter(
            [{"pt": 1.0, "eta": 1.0, "phi": 1.0, "mass": 1.0, "energy": 1.0}],
            "__list__",
            "OnlyAllowedFields",
            behavior=behavior,
        )
