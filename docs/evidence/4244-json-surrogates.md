# Issue #4244 evidence

- Issue: https://github.com/scikit-hep/awkward/issues/4244
- Baseline: `ak.to_json` returned `["\\udc80"]`; `ak.from_json` raised `ValueError`.
- Design: reject unpaired surrogates before JSON serialization.
- Regression: `tests/test_0019_use_json_library.py::test_chararray_rejects_unpaired_surrogates`.
- Verification: run the focused test and the JSON test module after the change.
- Review and merge: pending upstream pull request.
