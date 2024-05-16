from __future__ import annotations

import json

import pyarrow

from .pyarrow import AwkwardArrowType, to_awkwardarrow_storage_types

AWKWARD_INFO_KEY = b"awkward_info"  # metadata field in Table schema


def convert_awkward_arrow_table_to_native(aatable: pyarrow.Table) -> pyarrow.Table:
    """
    aatable: A pyarrow Table created with extensionarray=True
    returns: A pyarrow Table without extensionsarrays, but
      with 'awkward_info' in the schema's metadata that can be used to
      convert the resulting table back into one with extensionarrays.
    """
    new_fields = []
    metadata = []
    for aacol_field in aatable.schema:
        metadata.append(collect_ak_arr_type_metadata(aacol_field))
        new_field = awkward_arrow_field_to_native(aacol_field)
        new_fields.append(new_field)
    metadata_serial = json.dumps(metadata).encode(errors="surrogatescape")
    new_schema = pyarrow.schema(
        new_fields, metadata={AWKWARD_INFO_KEY: metadata_serial}
    )
    new_table = aatable.cast(new_schema)
    return new_table


def convert_native_arrow_table_to_awkward(table: pyarrow.Table) -> pyarrow.Table:
    """
    table: A pyarrow Table converted with convert_awkward_arrow_table_to_native
    returns: A pyarrow Table without extensionsarrays, but
      with 'awkward_info' in the schema's metadata that can be used to
      convert the resulting table back into one with extensionarrays.
    """
    new_fields = []
    metadata = json.loads(
        table.schema.metadata[AWKWARD_INFO_KEY].decode(errors="surrogatescape")
    )
    for aacol_field, field_metadata in zip(table.schema, metadata):
        new_fields.append(
            native_arrow_field_to_akarraytype(aacol_field, field_metadata)
        )
    new_schema = pyarrow.schema(new_fields, metadata=table.schema.metadata)
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
    metadata = typ._metadata_as_dict()
    metadata["field_name"] = aafield.name
    if typ.num_fields == 0:
        # Simple type
        return metadata
    # Compound type
    subfield_metadata_list = []
    for ifield in range(typ.num_fields):
        # Note: You can treat some, but not all, compound pyarrow types as iterators.
        # Note: AwkwardArrowType provides num_fields property but not field() method.
        ak_field = typ.storage_type.field(ifield)
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

    if typ.num_fields == 0:
        # We have a simple type wrapped in AwkwardArrowType.
        new_field = pyarrow.field(
            aafield.name, type=typ.storage_type, nullable=aafield.nullable
        )
        # print(f"  Returning simple field {new_field.name}: {new_field =}")
        return new_field

    # We have a container/compound type, wrapped in AwkwardArrowType.
    # print(f"field {aafield.name}")
    native_fields = []
    for ifield in range(typ.storage_type.num_fields):
        ak_field = typ.storage_type.field(ifield)
        # print(f"Sub-field {ak_field.name}: {ak_field}")
        native_fields.append(
            awkward_arrow_field_to_native(ak_field)  # Recurse
        )
    native_type = _make_pyarrow_type_like(typ, native_fields)
    new_field = pyarrow.field(aafield.name, type=native_type, nullable=aafield.nullable)
    # print(f"Returning new field {new_field.name}: {new_field}")
    return new_field


# TODO: add the remaining Arrow non-primitive types that we use
_pyarrow_type_builder = {
    pyarrow.lib.StructType: lambda *subfields: pyarrow.struct(subfields),
    pyarrow.lib.LargeListType: lambda subfield: pyarrow.large_list(subfield),
}


def native_arrow_field_to_akarraytype(
    ntv_field: pyarrow.Field, metadata: dict
) -> pyarrow.Field:
    if isinstance(ntv_field, AwkwardArrowType):
        raise ValueError(f"field {ntv_field} is already an AwkwardArrowType")
    storage_type = ntv_field.type

    if storage_type.num_fields > 0:
        # We need to replace storage_type with one that contains AwkwardArrowTypes.
        awkwardized_fields = []
        for ifield in range(storage_type.num_fields):
            subfield = storage_type.field(ifield)
            submeta = metadata["subfield_metadata"][ifield]
            awkwardized_fields.append(
                native_arrow_field_to_akarraytype(subfield, submeta)  # Recurse
            )
        storage_type = _make_pyarrow_type_like(storage_type, awkwardized_fields)

    ak_type = AwkwardArrowType._from_metadata_object(storage_type, metadata)
    return pyarrow.field(ntv_field.name, type=ak_type, nullable=ntv_field.nullable)


def _make_pyarrow_type_like(
    typ: pyarrow.Type, fields: list[pyarrow.Field]
) -> pyarrow.Type:
    storage_type = to_awkwardarrow_storage_types(typ)[1]
    if isinstance(storage_type, pyarrow.lib.DictionaryType):
        # TODO: num_fields == 0 but sub-types are value_type and index_type
        return pyarrow.dictionary(storage_type.index_type, storage_type.value_type)
    if isinstance(storage_type, pyarrow.lib.FixedSizeListType):
        return pyarrow.list_(fields[0], storage_type.list_size)
    if isinstance(storage_type, pyarrow.lib.ListType):
        return pyarrow.list_(fields[0])
    if isinstance(storage_type, pyarrow.lib.LargeListType):
        return pyarrow.large_list(fields[0])
    if isinstance(storage_type, pyarrow.lib.MapType):
        return pyarrow.map_(storage_type.index_type, fields[0])
    if isinstance(storage_type, pyarrow.lib.StructType):
        return pyarrow.struct(fields)
    if isinstance(storage_type, pyarrow.lib.UnionType):
        return pyarrow.union(fields, storage_type.mode, storage_type.type_codes)
    if isinstance(storage_type, pyarrow.lib.DataType) and storage_type.num_fields == 0:
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
        for col, new_field in zip(batch.itercolumns(), new_schema):
            columns.append(array_with_replacement_type(col, new_field.type))
        new_batches.append(
            pyarrow.RecordBatch.from_arrays(arrays=columns, schema=new_schema)
        )
    return pyarrow.Table.from_batches(new_batches)


def array_with_replacement_type(
    orig_array: pyarrow.Array, new_type: pyarrow.Type
) -> pyarrow.Array:
    children_orig = _get_children(orig_array)
    if len(children_orig) != new_type.num_fields:
        # Probable error in _get_children, not the passed arguments.
        raise AssertionError(
            f"Number of children: {len(children_orig) =} != {new_type.num_fields =}"
        )
    children_new = []
    for idx, child in enumerate(children_orig):
        new_child_type = new_type.field(idx).type
        children_new.append(array_with_replacement_type(child, new_child_type))
    own_buffers = orig_array.buffers()[: orig_array.type.num_buffers]
    return pyarrow.Array.from_buffers(
        type=new_type,
        length=len(orig_array),
        buffers=own_buffers,
        null_count=orig_array.null_count,
        offset=orig_array.offset,
        children=children_new,
    )


def _get_children(array: pyarrow.Array) -> list[pyarrow.Array]:
    arrow_type = to_awkwardarrow_storage_types(array.type)[1]
    if array.type.num_fields == 0:
        return []
    if hasattr(array, "field"):
        return [array.field(idx) for idx in range(array.type.num_fields)]
    if hasattr(array, "values"):
        return [
            array.values,
        ]
    raise NotImplementedError(f"Cannot get children of arrow type {arrow_type}")
