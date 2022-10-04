# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

_has_checked_version = False
_is_registered = False


def register_and_check():
    global _has_checked_version

    try:
        import numba
    except ImportError as err:
        raise ImportError(
            """install the 'numba' package with:

pip install numba --upgrade

or

conda install numba"""
        ) from err

    if not _has_checked_version:
        if ak._util.parse_version(numba.__version__) < ak._util.parse_version("0.50"):
            raise ImportError(
                "Awkward Array can only work with numba 0.50 or later "
                "(you have version {})".format(numba.__version__)
            )
        _has_checked_version = True

    _register()


def _register():
    if hasattr(ak.numba, "ArrayViewType"):
        return

    import numba

    import awkward._connect.numba.arrayview
    import awkward._connect.numba.builder
    import awkward._connect.numba.layout

    n = ak.numba
    n.ArrayViewType = awkward._connect.numba.arrayview.ArrayViewType
    n.ArrayViewModel = awkward._connect.numba.arrayview.ArrayViewModel
    n.RecordViewType = awkward._connect.numba.arrayview.RecordViewType
    n.RecordViewModel = awkward._connect.numba.arrayview.RecordViewModel
    n.ContentType = awkward._connect.numba.layout.ContentType
    n.NumpyArrayType = awkward._connect.numba.layout.NumpyArrayType
    n.RegularArrayType = awkward._connect.numba.layout.RegularArrayType
    n.ListArrayType = awkward._connect.numba.layout.ListArrayType
    n.IndexedArrayType = awkward._connect.numba.layout.IndexedArrayType
    n.IndexedOptionArrayType = awkward._connect.numba.layout.IndexedOptionArrayType
    n.ByteMaskedArrayType = awkward._connect.numba.layout.ByteMaskedArrayType
    n.BitMaskedArrayType = awkward._connect.numba.layout.BitMaskedArrayType
    n.UnmaskedArrayType = awkward._connect.numba.layout.UnmaskedArrayType
    n.RecordArrayType = awkward._connect.numba.layout.RecordArrayType
    n.UnionArrayType = awkward._connect.numba.layout.UnionArrayType
    n.ArrayBuilderType = awkward._connect.numba.builder.ArrayBuilderType
    n.ArrayBuilderModel = awkward._connect.numba.builder.ArrayBuilderModel

    @numba.extending.typeof_impl.register(ak.highlevel.Array)
    def typeof_Array(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak.highlevel.Record)
    def typeof_Record(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak.highlevel.ArrayBuilder)
    def typeof_ArrayBuilder(obj, c):
        return obj.numba_type
