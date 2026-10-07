# JSON output rejects unpaired surrogates

Issue: [#4244](https://github.com/scikit-hep/awkward/issues/4244)

`NumpyArray` character data uses Python's `surrogateescape` decoding so that
invalid UTF-8 bytes remain representable in an Awkward string. JSON text can
only contain Unicode scalar values, however, and `ak.from_json` rejects the
lone-surrogate escape that Python's JSON encoder emits for those values.

`ak.to_json` now validates strings produced by the layout before serializing
them and raises `ValueError` for an unpaired surrogate. This keeps the output
contract honest and avoids emitting text that the paired `ak.from_json`
operation cannot read. Replacing the byte with U+FFFD or inventing a JSON
escape would silently lose the original byte, so those alternatives are
rejected.
