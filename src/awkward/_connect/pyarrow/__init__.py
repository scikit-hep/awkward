# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from types import ModuleType

from packaging.version import parse as parse_version

__all__ = [
    "import_pyarrow",
    "import_pyarrow_parquet",
    "import_pyarrow_compute",
    "AwkwardArrowArray",
    "AwkwardArrowType",
    "and_validbytes",
    "convert_to_array",
    "direct_Content_subclass",
    "direct_Content_subclass_name",
    "form_handle_arrow",
    "handle_arrow",
    "popbuffers",
    "remove_optiontype",
    "to_awkwardarrow_storage_types",
    "to_awkwardarrow_type",
    "to_length",
    "to_null_count",
    "to_validbits",
    "convert_awkward_arrow_table_to_native",
    "convert_native_arrow_table_to_awkward",
]

try:
    import pyarrow

    error_message = None

except ModuleNotFoundError:
    pyarrow = None
    error_message = """to use {0}, you must install pyarrow:

    pip install pyarrow

or

    conda install -c conda-forge pyarrow
"""

else:
    if parse_version(pyarrow.__version__) < parse_version("7.0.0"):
        pyarrow = None
        error_message = "pyarrow 7.0.0 or later required for {0}"

if error_message is None:
    from .conversions import (
        and_validbytes,
        convert_to_array,
        direct_Content_subclass,
        direct_Content_subclass_name,
        form_handle_arrow,
        handle_arrow,
        popbuffers,
        remove_optiontype,
        to_awkwardarrow_type,
        to_length,
        to_null_count,
        to_validbits,
    )
    from .extn_types import (
        AwkwardArrowArray,
        AwkwardArrowType,
        to_awkwardarrow_storage_types,
    )
    from .table_conv import (
        convert_awkward_arrow_table_to_native,
        convert_native_arrow_table_to_awkward,
    )
else:
    AwkwardArrowArray = None
    AwkwardArrowType = None

    def nothing_without_pyarrow(*args, **kwargs):
        raise NotImplementedError(
            "This function requires pyarrow, which is not installed."
        )

    convert_awkward_arrow_table_to_native = nothing_without_pyarrow
    convert_native_arrow_table_to_awkward = nothing_without_pyarrow
    and_validbytes = nothing_without_pyarrow
    to_validbits = nothing_without_pyarrow
    to_length = nothing_without_pyarrow
    to_null_count = nothing_without_pyarrow
    to_awkwardarrow_storage_types = nothing_without_pyarrow
    popbuffers = nothing_without_pyarrow
    handle_arrow = nothing_without_pyarrow
    convert_to_array = nothing_without_pyarrow
    to_awkwardarrow_type = nothing_without_pyarrow
    direct_Content_subclass = nothing_without_pyarrow
    direct_Content_subclass_name = nothing_without_pyarrow
    remove_optiontype = nothing_without_pyarrow
    form_handle_arrow = nothing_without_pyarrow


def import_pyarrow(name: str) -> ModuleType:
    if pyarrow is None:
        raise ImportError(error_message.format(name))
    return pyarrow


def import_pyarrow_parquet(name: str) -> ModuleType:
    if pyarrow is None:
        raise ImportError(error_message.format(name))

    import pyarrow.parquet as out

    return out


def import_pyarrow_compute(name: str) -> ModuleType:
    if pyarrow is None:
        raise ImportError(error_message.format(name))

    import pyarrow.compute as out

    return out
