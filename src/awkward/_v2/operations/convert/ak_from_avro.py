# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# formfromavro: bool; showcode: bool; readerlang: bool;
def from_avro_file(
    file_name, show_code=False, reader_lang="ft"
):  # behavior and high level
    import awkward._v2._connect.avro

    if reader_lang == "py":
        temp_class = awkward._v2._connect.avro.read_avro_py(file_name, show_code)
        if show_code:
            print("".join(temp_class.head + temp_class.body))  # noqa
        return temp_class.outarr  # wrap the layout here using from_buffers.impl
    elif reader_lang == "ft":
        pass


def metadata_from_avro_file(file_name, show_code=False, reader_lang="ft"):
    import awkward._v2._connect.avro

    if reader_lang == "py":
        temp_class = awkward._v2._connect.avro.read_avro_py(file_name, show_code)
        if show_code:
            print("".join(temp_class.head + temp_class.body))  # noqa
        return temp_class.form
    elif reader_lang == "ft":
        pass
