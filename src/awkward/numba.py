# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


def register():
    import awkward._connect._numba
    import awkward._v2.numba

    awkward._connect._numba.register()
    awkward._v2.numba.register()
