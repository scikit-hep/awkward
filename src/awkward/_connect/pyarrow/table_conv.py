from __future__ import annotations

import json

import pyarrow

from .extn_types import (
    AwkwardArrowArray,
    AwkwardArrowType,
    to_awkwardarrow_storage_types,
)

AWKWARD_INFO_KEY = b"awkward_array_metadata"  # metadata field in Table schema


def convert_awkward_arrow_table_to_native(aatable: pyarrow.Table) -> pyarrow.Table:
    """
    aatable: A pyarrow Table created with extensionarray=True
    returns: A pyarrow Table without extensionsarrays, but
      with `awkward_array_metadata` in the schema's metadata that can be used to
      convert the resulting table back into one with extensionarrays.
    """
    new_fields = []
    metadata = {}  # metadata for table column types
    for aacol_field in aatable.schema:
        metadata[aacol_field.name] = collect_ak_arr_type_metadata(aacol_field)
        new_field = awkward_arrow_field_to_native(aacol_field)
        new_fields.append(new_field)
    metadata_serial = json.dumps(metadata).encode(errors="surrogatescape")
    if aatable.schema.metadata is None:
        new_metadata = {}
    else:
        new_metadata = aatable.schema.metadata.copy()
    new_metadata[AWKWARD_INFO_KEY] = metadata_serial
    new_schema = pyarrow.schema(new_fields, metadata=new_metadata)
    # return = aatable.cast(new_schema)
    return replace_schema(aatable, new_schema)


def convert_native_arrow_table_to_awkward(table: pyarrow.Table) -> pyarrow.Table:
    """
    table: A pyarrow Table converted with convert_awkward_arrow_table_to_native
    returns: A pyarrow Table without extensionsarrays, but
      with `awkward_array_metadata` in the schema's metadata that can be used to
      convert the resulting table back into one with extensionarrays.
    """
    if table.schema.metadata is None or AWKWARD_INFO_KEY not in table.schema.metadata:
        return table  # Prior versions don't include metadata here
    new_fields = []
    metadata = json.loads(
        table.schema.metadata[AWKWARD_INFO_KEY].decode(errors="surrogatescape")
    )
    for aacol_field in table.schema:
        if aacol_field.name not in metadata:
            raise ValueError(
                f"Awkward metadata in Arrow table does not have info for column {aacol_field.name}"
            )
        new_fields.append(
            native_arrow_field_to_akarraytype(aacol_field, metadata[aacol_field.name])
        )
    new_metadata = table.schema.metadata.copy()
    del new_metadata[AWKWARD_INFO_KEY]
    new_schema = pyarrow.schema(new_fields, metadata=new_metadata)
    # return table.cast(new_schema)  # Similar (same even?) results
    return replace_schema(table, new_schema)


def collect_ak_arr_type_metadata(aafield: pyarrow.Field) -> dict | list | None:
    """
    Given a Field, collect ArrowExtensionArray metadata as an object.
    If that field holds more ArrowExtensionArray types, a "subfield_metadata"
    property is added that holds a list of metadata objects for the sub-fields.
    This recurses down the whole type structure.
    """
    typ = aafield.type
    if not isinstance(typ, AwkwardArrowType):
        return None  # Not expected to reach here
    subfields = _fields_of_strg_type(typ.storage_type)
    metadata = typ._metadata_as_dict()
    metadata["field_name"] = aafield.name
    if len(subfields) == 0:
        # Simple type
        return metadata
    # Compound type
    subfield_metadata_list = []
    for ak_field in subfields:
        subfield_metadata_list.append(
            collect_ak_arr_type_metadata(ak_field)  # Recurse
        )
    metadata["subfield_metadata"] = subfield_metadata_list
    return metadata


def awkward_arrow_field_to_native(aafield: pyarrow.Field) -> pyarrow.Field:
    """
    Given a Field with ArrowExtensionArray type, returns a corresponding
    field with only Arrow builtin, or storage, types. Metadata is removed.
    """
    typ = aafield.type
    if not isinstance(typ, AwkwardArrowType):
        # Not expected to reach this. Maybe throw ValueError?
        return aafield

    fields = _fields_of_strg_type(typ.storage_type)
    if len(fields) == 0:
        # We have a simple type wrapped in AwkwardArrowType.
        new_field = pyarrow.field(
            aafield.name, type=typ.storage_type, nullable=aafield.nullable
        )
        return new_field

    # We have nested types
    native_fields = [
        awkward_arrow_field_to_native(field)  # Recurse
        for field in fields
    ]
    native_type = _make_pyarrow_type_like(typ, native_fields)
    new_field = pyarrow.field(aafield.name, type=native_type, nullable=aafield.nullable)
    return new_field


def native_arrow_field_to_akarraytype(
    ntv_field: pyarrow.Field, metadata: dict
) -> pyarrow.Field:
    if isinstance(ntv_field, AwkwardArrowType):
        raise ValueError(f"field {ntv_field} is already an AwkwardArrowType")
    storage_type = ntv_field.type
    fields = _fields_of_strg_type(storage_type)
    if len(fields) > 0:
        # We need to replace storage_type with one that contains AwkwardArrowTypes.
        awkwardized_fields = [
            native_arrow_field_to_akarraytype(field, meta)  # Recurse
            for field, meta in zip(fields, metadata["subfield_metadata"])
        ]
        storage_type = _make_pyarrow_type_like(storage_type, awkwardized_fields)
    ak_type = AwkwardArrowType._from_metadata_object(storage_type, metadata)
    return pyarrow.field(ntv_field.name, type=ak_type, nullable=ntv_field.nullable)


def _fields_of_strg_type(typ: pyarrow.Type) -> list[pyarrow.Field]:
    if isinstance(typ, pyarrow.lib.DictionaryType):
        return [
            pyarrow.field("value", typ.value_type)
        ]  # Wrap in a field for consistency
    elif typ.num_fields == 0:
        return []
    elif not hasattr(typ, "field"):
        # Old versions of pyarrow have this quirk.
        if hasattr(typ, "value_field"):
            return [typ.value_field]
        elif hasattr(typ, "__iter__"):
            return list(typ)
        raise TypeError(f"Cannot handle arrow type {typ}")
    else:
        return [typ.field(i) for i in range(typ.num_fields)]


def _make_pyarrow_type_like(
    typ: pyarrow.Type, fields: list[pyarrow.Field]
) -> pyarrow.Type:
    storage_type = to_awkwardarrow_storage_types(typ)[1]
    if isinstance(storage_type, pyarrow.lib.DictionaryType):
        return pyarrow.dictionary(storage_type.index_type, fields[0].type)
    elif isinstance(storage_type, pyarrow.lib.FixedSizeListType):
        return pyarrow.list_(fields[0], storage_type.list_size)
    elif isinstance(storage_type, pyarrow.lib.ListType):
        return pyarrow.list_(fields[0])
    elif isinstance(storage_type, pyarrow.lib.LargeListType):
        return pyarrow.large_list(fields[0])
    elif isinstance(storage_type, pyarrow.lib.MapType):
        # return pyarrow.map_(storage_type.index_type, fields[0])
        raise NotImplementedError("pyarrow MapType is not supported by Awkward")
    elif isinstance(storage_type, pyarrow.lib.StructType):
        return pyarrow.struct(fields)
    elif isinstance(storage_type, pyarrow.lib.UnionType):
        return pyarrow.union(fields, storage_type.mode, storage_type.type_codes)
    elif (
        isinstance(storage_type, pyarrow.lib.DataType) and storage_type.num_fields == 0
    ):
        # Catch-all for primitive types, nulltype, string types, FixedSizeBinaryType
        return storage_type
    raise NotImplementedError(f"Type {typ} is not handled for conversion.")


def replace_schema(table: pyarrow.Table, new_schema: pyarrow.Schema) -> pyarrow.Table:
    """
    This function is like `pyarrow.Table.cast()` except it only works if the
    new schema uses the same storage types and storage geometries as the original.
    It explicitly will not convert one primitive type to another.
    """
    new_batches = []
    for batch in table.to_batches():
        columns = []
        for col, new_field in zip(batch.columns, new_schema):
            columns.append(array_with_replacement_type(col, new_field.type))
        new_batches.append(
            pyarrow.RecordBatch.from_arrays(arrays=columns, schema=new_schema)
        )
    return pyarrow.Table.from_batches(new_batches)


def array_with_replacement_type(
    orig_array: pyarrow.Array, new_type: pyarrow.Type
) -> pyarrow.Array:
    """
    Creates a new array with a different type.
    Either pyarrow native -> ExtensionArray or vice-versa.
    """
    children_orig = _get_children(orig_array)
    native_type = to_awkwardarrow_storage_types(new_type)[1]
    new_fields = _fields_of_strg_type(native_type)
    if len(new_fields) != len(children_orig):
        raise AssertionError(
            f"Number of children: {len(children_orig) =} != {len(new_fields) =}"
        )
    children_new = [
        array_with_replacement_type(child, new_child_type.type)
        for child, new_child_type in zip(children_orig, new_fields)
    ]
    own_buffers = orig_array.buffers()[: orig_array.type.num_buffers]
    if isinstance(native_type, pyarrow.lib.DictionaryType):
        # The following works with newer pyarrow versions, but not 7.0:
        # native_dict = pyarrow.DictionaryArray.from_buffers(
        #     type=native_type,
        #     length=len(orig_array),
        #     buffers=own_buffers,
        #     dictionary=children_new[0],
        #     null_count=orig_array.null_count,
        #     offset=orig_array.offset,
        # )
        if isinstance(orig_array, AwkwardArrowArray):
            native_orig = orig_array.storage
        else:
            native_orig = orig_array
        native_dict = pyarrow.DictionaryArray.from_arrays(
            indices=native_orig.indices,
            dictionary=children_new[0],
            mask=own_buffers[0],
            safe=False,
        )
        if isinstance(new_type, pyarrow.ExtensionType):
            return AwkwardArrowArray.from_storage(new_type, native_dict)
        else:
            return native_dict
    else:
        return pyarrow.Array.from_buffers(
            type=new_type,
            length=len(orig_array),
            buffers=own_buffers,
            null_count=orig_array.null_count,
            offset=orig_array.offset,
            children=children_new,
        )


def _get_children(array: pyarrow.Array) -> list[pyarrow.Array]:
    """
    Different types of pyarrow Arrays have different ways to
    access their "children." It helps to unify these.
    """
    arrow_type = to_awkwardarrow_storage_types(array.type)[1]
    if isinstance(array, AwkwardArrowArray):
        array = array.storage

    if isinstance(arrow_type, pyarrow.lib.DictionaryType):
        return [array.dictionary]
    if arrow_type.num_fields == 0:
        return []
    if hasattr(array, "field"):
        return [array.field(idx) for idx in range(arrow_type.num_fields)]
    if hasattr(array, "values"):
        return [array.values]
    raise NotImplementedError(f"Cannot get children of arrow type {arrow_type}")
