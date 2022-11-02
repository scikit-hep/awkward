# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

raise ModuleNotFoundError(  # noqa: AK101
    """The awkward._v2 submodule was provided for early access to awkward>=2, as it developed.

Now that version 2 has been released, awkward._v2 is no longer needed.

If you were an early adopter using

    import awkward._v2 as ak

you can replace it with

    import awkward as ak

or

    try:
        import awkward._v2 as ak   # provides v2 in 1.8.0rc1<=awkward<=1.10.1
    except ModuleNotFoundError:
        import awkward as ak       # provides v2 in awkward>=2

It is no longer possible to access v1 and v2 in the same process.

Arrays can be written and read as Parquet files or Arrow buffers to share data between processes."""
)
