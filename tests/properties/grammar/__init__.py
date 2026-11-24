# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for grammars, one module per grammar file.

Each module generates strings from a grammar with
`hypothesis.extra.lark.from_lark` and asserts that awkward accepts
every one. Such a test finds strings the grammar allows but awkward
rejects; it does not find strings awkward accepts but the grammar
lacks.

A grammar should describe exactly what awkward accepts, so a module
has no predicate for deliberate rejections. Every known cause of a
rejection has a private predicate, named after what it detects and
called from `_would_raise_from_known_issue`. Delete a predicate in
the pull request that removes its cause: the skipped strings are then
tested again.
"""
