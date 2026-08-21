# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for awkward operations, one module per operation.

The operation modules (`test_flatten.py`, `test_all.py`,
`test_ravel.py`) follow one template, and each is a complete example
to copy. The identifiers
are operation-invariant — only the module name and the called
operation say what is under test. `Kwargs`, `DEFAULTS`, and
`RelatedKwargs` declare the operation's options;
`test_kwargs_match_signature` checks them against the signature
(through `util`); `st_kwargs` draws an option dict in which every
option is optional, so the defaults are exercised too; and
`test_properties`, the single property test, draws an array and
options, keeps only the draws expected to succeed, and calls the
operation.

Two predicates decide which draws are kept. `_should_not_raise`
models the design — what the operation deliberately accepts and
refuses — and is re-derived for each operation.
`_would_raise_from_known_issue` skips draws that meet an open issue;
its clauses select predicates from `known_issues` and are deleted as
the issues are fixed. Both are conservative, and their safe
directions differ: a needlessly skipped draw costs only coverage
while a missing skip fails the suite, so narrow `_should_not_raise`
toward `False` and widen a `known_issues` trigger.

The `test_to_from_*` and `test_array_equal` modules are an earlier
family with their own pattern.
"""
