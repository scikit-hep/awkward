# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# from awkward.typing import Type
import pathlib

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def from_avro_file(
    file, limit_entries=None, *, debug_forth=False, highlevel=True, behavior=None
):
    """
    Args:
        file (string or file-like object): Avro file to be read as Awkward Array.
        limit_entries (int): The number of rows of the Avro file to be read into the Awkward Array.
        debug_forth (bool): If True, prints the generated Forth code for debugging.
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
            limit_entries=limit_entries,
            debug_forth=debug_forth,
        ),
    ):

        if isinstance(file, pathlib.Path):
            file = str(file)

        if isinstance(file, str):
            with open(file, "rb") as opened_file:
                form, length, container = awkward._connect.avro.ReadAvroFT(
                    opened_file, limit_entries, debug_forth
                ).outcontents
                return _impl(form, length, container, highlevel, behavior)

        else:
            if not hasattr(file, "read") or not hasattr(file, "seek"):
                raise ak._errors.wrap_error(
                    TypeError(
                        "'file' must either be a filename string or be a file-like object with 'read' and 'seek' methods"
                    )
                )
            else:
                form, length, container = awkward._connect.avro.ReadAvroFT(
                    file, limit_entries, debug_forth
                ).outarr
                return _impl(form, length, container, highlevel, behavior)


def _impl(form, length, container, highlevel, behavior):

    return ak.operations.ak_from_buffers._impl(
        form=form,
        length=length,
        container=container,
        buffer_key="{form_key}-{attribute}",
        backend="cpu",
        highlevel=highlevel,
        behavior=behavior,
        simplify=True,
    )
