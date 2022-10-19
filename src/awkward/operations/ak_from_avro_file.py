# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# from typing import Type
import pathlib

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


def from_avro_file(
    file, debug_forth=False, limit_entries=None, highlevel=True, behavior=None
):
    """
    Args:
        file (string or fileobject): Avro file to be read as Awkward Array.
        debug_forth (bool): If True, prints the generated Forth code for debugging.
        limit_entries (int): The number of rows of the Avro file to be read into the Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
    Reads Avro files as Awkward Arrays.

    Internally this function uses AwkwardForth DSL. The function recursively parses the Avro schema, generates
    Awkward form and Forth code for that specific Avro file and then reads it.
    """
    import awkward._connect.avro

    with ak._errors.OperationErrorContext(
        "ak.from_avro_file",
        dict(
            file=file,
            highlevel=highlevel,
            behavior=behavior,
            debug_forth=debug_forth,
            limit_entries=limit_entries,
        ),
    ):

        if isinstance(file, pathlib.Path):
            file = str(file)

        if isinstance(file, str):
            try:
                with open(file, "rb") as opened_file:
                    form, length, container = awkward._connect.avro.ReadAvroFT(
                        opened_file, limit_entries, debug_forth
                    ).outcontents
                    return _impl(form, length, container, highlevel, behavior)
            except ImportError as err:
                raise ak._errors.wrap_error(
                    "the filename is incorrect or the file does not exist"
                ) from err

        else:
            if not hasattr(file, "read"):
                raise ak._errors.wrap_error(
                    TypeError("the fileobject provided is not of the correct type.")
                )
            else:
                form, length, container = awkward._connect.avro.ReadAvroFT(
                    file, limit_entries, debug_forth
                ).outarr
                return _impl(form, length, container, highlevel, behavior)


def _impl(form, length, container, highlevel, behavior):

    return ak.from_buffers(
        form=form,
        length=length,
        container=container,
        highlevel=highlevel,
        behavior=behavior,
    )
