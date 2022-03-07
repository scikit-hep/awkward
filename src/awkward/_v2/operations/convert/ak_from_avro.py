# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_avro(file_name, read_data=True):
    import awkward._v2._connect.avro

    temp_class = awkward._v2._connect.avro.read_avro(
        file_name
    )
    return ak._v2._util.wrap(temp_class.outarr)
