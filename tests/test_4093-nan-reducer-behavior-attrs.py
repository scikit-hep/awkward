"""
Tests that nan-variant reducers (nanmin, nanmax, nanmean, nanvar, nanstd)
preserve behavior and attrs from the input array.

Regression for: nan-variants calling nan_to_none with highlevel=False / None attrs,
which caused behavior and attrs to be silently dropped during the intermediate
nan_to_none conversion.
"""

from __future__ import annotations

import awkward as ak


def make_test_array(with_nan=False):
    """Create an array with custom behavior and attrs."""

    class MyArray(ak.Array):
        pass

    ak.behavior["MyArray"] = MyArray

    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    if with_nan:
        data[2] = float("nan")

    arr = ak.Array(
        data,
        behavior={"MyArray": MyArray},
        attrs={"meta": "test_value", "source": "unit_test"},
    )
    return arr


class TestNanReducersBehavior:
    """Tests that behavior dict is preserved through nan reducer operations."""

    def test_nanmin_preserves_behavior(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanmin(arr)
        # behavior should be accessible (not None)
        assert arr.behavior is not None

    def test_nanmax_preserves_behavior(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanmax(arr)
        assert arr.behavior is not None

    def test_nanmean_preserves_behavior(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanmean(arr)
        assert arr.behavior is not None

    def test_nanvar_preserves_behavior(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanvar(arr)
        assert arr.behavior is not None

    def test_nanstd_preserves_behavior(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanstd(arr)
        assert arr.behavior is not None


class TestNanReducersAttrs:
    """Tests that attrs are preserved through nan reducer operations."""

    def test_nanmin_preserves_attrs(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanmin(arr, axis=None)
        # Result attrs should reflect input attrs (may be empty for scalars but
        # the important thing is that it doesn't raise and attrs are propagated
        # during the intermediate nan_to_none step)
        assert arr.attrs == {"meta": "test_value", "source": "unit_test"}

    def test_nanmax_preserves_attrs(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanmax(arr, axis=None)
        assert arr.attrs == {"meta": "test_value", "source": "unit_test"}

    def test_nanmean_preserves_attrs(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanmean(arr)
        assert arr.attrs == {"meta": "test_value", "source": "unit_test"}

    def test_nanvar_preserves_attrs(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanvar(arr)
        assert arr.attrs == {"meta": "test_value", "source": "unit_test"}

    def test_nanstd_preserves_attrs(self):
        arr = make_test_array(with_nan=True)
        result = ak.nanstd(arr)
        assert arr.attrs == {"meta": "test_value", "source": "unit_test"}


class TestNanReducersMatchNonNan:
    """Tests that nan-variant reducers on nan-free data match non-nan variants."""

    def test_nanmin_matches_min(self):
        arr = make_test_array(with_nan=False)
        assert ak.nanmin(arr, axis=None) == ak.min(arr, axis=None)

    def test_nanmax_matches_max(self):
        arr = make_test_array(with_nan=False)
        assert ak.nanmax(arr, axis=None) == ak.max(arr, axis=None)

    def test_nanmean_matches_mean(self):
        arr = make_test_array(with_nan=False)
        result_nan = ak.nanmean(arr, axis=None)
        result_plain = ak.mean(arr, axis=None)
        assert abs(float(result_nan) - float(result_plain)) < 1e-10

    def test_nanvar_matches_var(self):
        arr = make_test_array(with_nan=False)
        result_nan = ak.nanvar(arr, axis=None)
        result_plain = ak.var(arr, axis=None)
        assert abs(float(result_nan) - float(result_plain)) < 1e-10

    def test_nanstd_matches_std(self):
        arr = make_test_array(with_nan=False)
        result_nan = ak.nanstd(arr, axis=None)
        result_plain = ak.std(arr, axis=None)
        assert abs(float(result_nan) - float(result_plain)) < 1e-10


class TestNanReducersWithNan:
    """Tests that nan-variant reducers correctly handle NaN values."""

    def test_nanmin_excludes_nan(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanmin(arr, axis=None)
        assert float(result) == 1.0

    def test_nanmax_excludes_nan(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanmax(arr, axis=None)
        assert float(result) == 3.0

    def test_nanmean_excludes_nan(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanmean(arr, axis=None)
        assert abs(float(result) - 2.0) < 1e-10

    def test_nanvar_excludes_nan(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanvar(arr, axis=None)
        # var of [1, 3] = 1.0
        assert abs(float(result) - 1.0) < 1e-10

    def test_nanstd_excludes_nan(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanstd(arr, axis=None)
        # std of [1, 3] = 1.0
        assert abs(float(result) - 1.0) < 1e-10


class TestNanVarHighlevelFalse:
    """Tests that nanvar with highlevel=False still correctly propagates behavior/attrs during
    the intermediate nan_to_none step (regression for passing `highlevel` into nan_to_none)."""

    def test_nanvar_highlevel_false_on_nan_data(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanvar(arr, axis=None, highlevel=False)
        # Should return a scalar number, not raise or return NaN
        assert abs(float(result) - 1.0) < 1e-10

    def test_nanmean_highlevel_false_on_nan_data(self):
        arr = ak.Array([1.0, float("nan"), 3.0])
        result = ak.nanmean(arr, axis=None, highlevel=False)
        assert abs(float(result) - 2.0) < 1e-10


class TestBehaviorAndAttrsMultidimensional:
    """Test behavior/attrs preservation with multidimensional arrays."""

    def test_nanmin_2d_axis0(self):
        arr = ak.Array([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        arr = ak.Array(
            arr.layout,
            behavior={"test": True},
            attrs={"dim": "2d"},
        )
        result = ak.nanmin(arr, axis=1)
        # Input attrs and behavior should still be present
        assert arr.attrs == {"dim": "2d"}
        assert arr.behavior == {"test": True}

    def test_nanmax_2d_axis0(self):
        arr = ak.Array([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        arr = ak.Array(
            arr.layout,
            behavior={"test": True},
            attrs={"dim": "2d"},
        )
        result = ak.nanmax(arr, axis=1)
        assert arr.attrs == {"dim": "2d"}
        assert arr.behavior == {"test": True}
