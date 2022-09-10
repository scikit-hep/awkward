# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

checked_version = False


def register_and_check():
    global checked_version

    if not checked_version:
        try:
            import numba
        except ImportError as err:
            raise ImportError(
                """install the 'numba' package with:

    pip install numba --upgrade

or

    conda install numba"""
            ) from err

        checked_version = True
        if ak._v2._util.parse_version(numba.__version__) < ak._v2._util.parse_version(
            "0.50"
        ):
            raise ImportError(
                "Awkward Array can only work with numba 0.50 or later "
                "(you have version {})".format(numba.__version__)
            )

    register()


def register():
    if hasattr(ak._v2.numba, "ArrayViewType"):
        return

    import numba
    import awkward._v2._connect.numba.arrayview
    import awkward._v2._connect.numba.layout
    import awkward._v2._connect.numba.builder

    n = ak._v2.numba
    n.ArrayViewType = awkward._v2._connect.numba.arrayview.ArrayViewType
    n.ArrayViewModel = awkward._v2._connect.numba.arrayview.ArrayViewModel
    n.RecordViewType = awkward._v2._connect.numba.arrayview.RecordViewType
    n.RecordViewModel = awkward._v2._connect.numba.arrayview.RecordViewModel
    n.ContentType = awkward._v2._connect.numba.layout.ContentType
    n.NumpyArrayType = awkward._v2._connect.numba.layout.NumpyArrayType
    n.RegularArrayType = awkward._v2._connect.numba.layout.RegularArrayType
    n.ListArrayType = awkward._v2._connect.numba.layout.ListArrayType
    n.IndexedArrayType = awkward._v2._connect.numba.layout.IndexedArrayType
    n.IndexedOptionArrayType = awkward._v2._connect.numba.layout.IndexedOptionArrayType
    n.ByteMaskedArrayType = awkward._v2._connect.numba.layout.ByteMaskedArrayType
    n.BitMaskedArrayType = awkward._v2._connect.numba.layout.BitMaskedArrayType
    n.UnmaskedArrayType = awkward._v2._connect.numba.layout.UnmaskedArrayType
    n.RecordArrayType = awkward._v2._connect.numba.layout.RecordArrayType
    n.UnionArrayType = awkward._v2._connect.numba.layout.UnionArrayType
    n.ArrayBuilderType = awkward._v2._connect.numba.builder.ArrayBuilderType
    n.ArrayBuilderModel = awkward._v2._connect.numba.builder.ArrayBuilderModel

    @numba.extending.typeof_impl.register(ak._v2.highlevel.Array)
    def typeof_Array(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak._v2.highlevel.Record)
    def typeof_Record(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak._v2.highlevel.ArrayBuilder)
    def typeof_ArrayBuilder(obj, c):
        return obj.numba_type
