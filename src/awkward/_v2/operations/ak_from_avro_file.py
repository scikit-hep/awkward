# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# from typing import Type
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# TODO: add OperationErrorContext; see ak_from_iter
# TODO: pathlib.Path is as good as str (ak._v2._util.regularize_path)
# TODO: limit_entries as a parameter would make metadata_from_avro_file unnecessary
# TODO: show_code => debug_forth


def from_avro_file(file, show_code=False):  # TODO: behavior and highlevel
    """
    TODO: doc string
    """
    import awkward._v2._connect.avro

    if isinstance(file, str):
        try:
            with open(file, "rb") as opened_file:
                return awkward._v2._connect.avro.ReadAvroFT(
                    opened_file, show_code
                ).outarr
        except ImportError:
            raise ImportError("Incorrect filename")

    else:
        if not hasattr(file, "read"):
            raise ak._v2._util.error(TypeError("The filetype is not correct"))
        else:
            return awkward._v2._connect.avro.ReadAvroFT(file, show_code).outarr
