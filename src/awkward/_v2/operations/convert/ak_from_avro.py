# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# from typing import Type
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# formfromavro: bool; showcode: bool; readerlang: bool;
def from_avro_file(file, show_code=False, reader_lang="ft"):  # behavior and high level
    import awkward._v2._connect.avro

    if isinstance(file, str):
        try:
            temp_name = file
            with open(file, "rb") as file:
                if reader_lang == "py":
                    temp_class = awkward._v2._connect.avro.read_avro_py(
                        temp_name, show_code
                    )
                    return (
                        temp_class.outarr
                    )  # wrap the layout here using from_buffers.impl
                elif reader_lang == "ft":
                    pass
                    temp_class = awkward._v2._connect.avro.ReadAvroFT(file, show_code)
                    return temp_class.outarr
        except ImportError:
            raise ImportError("Incorrect filename")
    else:
        if not hasattr(file, "read"):
            raise TypeError("The filetype is not correct")
        else:
            if reader_lang == "py":
                raise TypeError  # wrap the layout here using from_buffers.impl
            elif reader_lang == "ft":
                pass
                temp_class = awkward._v2._connect.avro.ReadAvroFT(file, show_code)
                return temp_class.outarr


# wrap the layout here using from_buffers.impl


def metadata_from_avro_file(file, show_code=False, reader_lang="ft"):
    import awkward._v2._connect.avro

    if reader_lang == "py":
        temp_class = awkward._v2._connect.avro.read_avro_py(file, show_code)
        return temp_class.form
    elif reader_lang == "ft":
        temp_class = awkward._v2._connect.avro.ReadAvroFT(file, show_code)
        return temp_class.form
