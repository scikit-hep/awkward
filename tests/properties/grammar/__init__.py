# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for grammars.

For each grammar file, a shared module, such as `datashape.py`, holds
the grammar and the predicates. Each test module tests one property
of the strings generated from the grammar with
`hypothesis.extra.lark.from_lark` and is named after it:
`test_parsable.py` asserts that awkward parses every string, and
`test_roundtrip.py` that the type read from a string prints and reads
back. `test_parsable.py` finds strings the grammar allows but awkward
rejects, not strings awkward accepts but the grammar lacks.
`test_roundtrip.py` also finds types that awkward prints as strings
the grammar lacks.

A grammar should describe exactly what awkward accepts, so there is
no predicate for deliberate rejections. Every known cause of a
failure has a public predicate in the shared module, named after what
it detects, and a test module calls the predicates it needs from a
private function, such as `_would_raise_from_known_issue`. Delete a
predicate in the pull request that removes its cause: the skipped
strings are then tested again.
"""
