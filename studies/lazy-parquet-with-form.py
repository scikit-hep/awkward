import json

import numpy as np
import awkward as ak
import pyarrow.parquet

_from_arrow = ak.operations.convert._from_arrow
_from_parquet_key = ak.operations.convert._from_parquet_key


def _parquet_schema_to_form(schema):
    def lst(path):
        return "lst:" + ".".join(path)

    def col(path):
        return "col:" + ".".join(path)

    def maybe_nullable(field, content):
        if field.nullable:
            return ak.forms.ByteMaskedForm(
                "i8",
                content.with_form_key(None),
                valid_when=True,
                form_key=content.form_key,
            )
        else:
            return content

    def contains_record(form):
        if isinstance(form, ak.forms.RecordForm):
            return True
        elif isinstance(form, ak.forms.ListOffsetForm):
            return contains_record(form.content)
        else:
            return False

    def recurse(arrow_type, path):
        if isinstance(arrow_type, pyarrow.StructType):
            names = []
            contents = []
            for index in range(arrow_type.num_fields):
                field = arrow_type[index]
                names.append(field.name)
                content = maybe_nullable(
                    field, recurse(field.type, path + (field.name,))
                )
                contents.append(ak.forms.VirtualForm(content, has_length=True))
            assert len(contents) != 0
            return ak.forms.RecordForm(contents, names)

        elif isinstance(arrow_type, pyarrow.ListType):
            field = arrow_type.value_field
            content = maybe_nullable(
                field, recurse(field.type, path + ("list", "item"))
            )
            form_key = None if contains_record(content) else lst(path)
            return ak.forms.ListOffsetForm("i32", content, form_key=form_key)

        elif isinstance(arrow_type, pyarrow.LargeListType):
            field = arrow_type.value_field
            content = maybe_nullable(
                field, recurse(field.type, path + ("list", "item"))
            )
            form_key = None if contains_record(content) else lst(path)
            return ak.forms.ListOffsetForm("i64", content, form_key=form_key)

        elif arrow_type == pyarrow.string():
            return ak.forms.ListOffsetForm(
                "i32",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
                form_key=col(path),
            )

        elif arrow_type == pyarrow.large_string():
            return ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
                form_key=col(path),
            )

        elif arrow_type == pyarrow.binary():
            return ak.forms.ListOffsetForm(
                "i32",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
                form_key=col(path),
            )

        elif arrow_type == pyarrow.large_binary():
            return ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
                form_key=col(path),
            )

        elif isinstance(arrow_type, pyarrow.DataType):
            dtype = np.dtype(arrow_type.to_pandas_dtype())
            return ak.forms.Form.from_numpy(dtype).with_form_key(col(path))

        else:
            raise NotImplementedError(
                "cannot convert {0}.{1} to an equivalent Awkward Form".format(
                    type(arrow_type).__module__, type(arrow_type).__name__
                )
            )

    schema = schema.to_arrow_schema()
    contents = []
    for index, name in enumerate(schema.names):
        field = schema.field(index)
        content = maybe_nullable(field, recurse(field.type, (name,)))
        contents.append(ak.forms.VirtualForm(content, has_length=True))
    assert len(contents) != 0
    return ak.forms.RecordForm(contents, schema.names)


class AlgebraicEffect(Exception):
    def __init__(self, state):
        self.state = state

    def __call__(self):
        globals().update(self.state)


class _ParquetState(object):
    def __init__(self, file):
        self.file = file

    def __call__(self, row_group, unpack, length, form, lazy_cache, lazy_cache_key):
        if form.form_key is None:
            if isinstance(form, ak.forms.RecordForm):
                contents = []
                recordlookup = []
                for i in range(form.numfields):
                    name = form.key(i)
                    subform = form.content(i).form
                    generator = ak.layout.ArrayGenerator(
                        self,
                        (
                            row_group,
                            unpack + (name,),
                            length,
                            subform,
                            lazy_cache,
                            lazy_cache_key,
                        ),
                        length=length,
                        form=subform,
                    )
                    if subform.form_key is None:
                        field_cache = None
                        cache_key = None
                    else:
                        field_cache = lazy_cache
                        cache_key = "{0}:{1}[{2}]".format(
                            lazy_cache_key, subform.form_key, row_group
                        )
                    contents.append(
                        ak.layout.VirtualArray(generator, field_cache, cache_key)
                    )
                    recordlookup.append(name)
                return ak.layout.RecordArray(contents, recordlookup, length)

            elif isinstance(form, ak.forms.ListOffsetForm):
                struct_only = [x for x in unpack[:0:-1] if x is not None]
                sampleform = _ParquetState_first_column(form, struct_only)
                sample = self.get(row_group, unpack, sampleform, struct_only)

                offsets = [sample.offsets]
                sublength = offsets[-1][-1]
                sample = sample.content
                recordform = form.content
                unpack = unpack + (None,)
                while not isinstance(recordform, ak.forms.RecordForm):
                    offsets.append(sample.offsets)
                    sublength = offsets[-1][-1]
                    sample = sample.content
                    recordform = recordform.content
                    unpack = unpack + (None,)

                out = self(
                    row_group, unpack, sublength, recordform, lazy_cache, lazy_cache_key
                )
                for off in offsets[::-1]:
                    if isinstance(off, ak.layout.Index32):
                        out = ak.layout.ListOffsetArray32(off, out)
                    elif isinstance(off, ak.layout.Index64):
                        out = ak.layout.ListOffsetArray64(off, out)
                    else:
                        raise AssertionError("unexpected Index type: {0}".format(off))
                return out

            else:
                raise AssertionError("unexpected Form: {0}".format(type(form)))

        else:
            assert form.form_key.startswith("col:") or form.form_key.startswith("lst:")
            column_name = form.form_key[4:]
            masked = isinstance(form, ak.forms.ByteMaskedForm)
            if masked:
                form = form.content
            table = self.file.read_row_group(row_group, [column_name])
            struct_only = [column_name.split(".")[-1]]
            struct_only.extend([x for x in unpack[:0:-1] if x is not None])
            return _ParquetState_arrow_to_awkward(table, struct_only, masked, unpack)

    def get(self, row_group, unpack, form, struct_only):
        assert form.form_key.startswith("col:") or form.form_key.startswith("lst:")
        column_name = form.form_key[4:]
        masked = isinstance(form, ak.forms.ByteMaskedForm)
        if masked:
            form = form.content
        table = self.file.read_row_group(row_group, [column_name])
        return _ParquetState_arrow_to_awkward(table, struct_only, masked, unpack)


def _ParquetState_first_column(form, struct_only):
    if isinstance(form, ak.forms.VirtualForm):
        return _ParquetState_first_column(form.form, struct_only)
    elif isinstance(form, ak.forms.RecordForm):
        assert form.numfields != 0
        struct_only.insert(0, form.key(0))
        return _ParquetState_first_column(form.content(0), struct_only)
    elif isinstance(form, ak.forms.ListOffsetForm):
        return _ParquetState_first_column(form.content, struct_only)
    else:
        return form


def _ParquetState_arrow_to_awkward(table, struct_only, masked, unpack):
    out = _from_arrow(table, False, struct_only=struct_only, highlevel=False)
    for item in unpack:
        if item is None:
            out = out.content
        else:
            out = out.field(item)
    if masked and not isinstance(out, ak.layout.ByteMaskedArray):
        out = out.toByteMaskedArray()
    return out


def from_parquet(source):
    file = pyarrow.parquet.ParquetFile(source)
    form = _parquet_schema_to_form(file.schema)
    all_columns = form.keys()
    columns = all_columns

    length = file.metadata.row_group(0).num_rows

    cache = {}
    hold_cache = ak._util.MappingProxy.maybe_wrap(cache)
    lazy_cache = ak.layout.ArrayCache(hold_cache)
    state = _ParquetState(file)

    lazy_cache_key = None
    if lazy_cache_key is None:
        lazy_cache_key = "ak.from_parquet:{0}".format(_from_parquet_key())

    row_group = 0
    fields = []
    names = []
    for column in columns:
        subform = form.contents[column].form
        generator = ak.layout.ArrayGenerator(
            state,
            (
                row_group,
                (column,),
                length,
                subform,
                lazy_cache,
                lazy_cache_key,
            ),
            length=length,
            form=form.contents[column].form,
        )
        if subform.form_key is None:
            field_cache = None
            cache_key = None
        else:
            field_cache = lazy_cache
            cache_key = "{0}:{1}[{2}]".format(
                lazy_cache_key, subform.form_key, row_group
            )
        fields.append(ak.layout.VirtualArray(generator, field_cache, cache_key))
        names.append(column)

    return ak.Array(ak.layout.RecordArray(fields, names))


ak.to_parquet(
    ak.Array(
        [
    {"x": [{"y": {"q": 1}, "z": 1.1}]},
    {"x": [{"y": {"q": 1}, "z": 1.1}, {"y": {"q": 2}, "z": 2.2}]},
    {"x": [{"y": {"q": 1}, "z": 1.1}, {"y": {"q": 2}, "z": 2.2}, {"y": {"q": 3}, "z": 3.3}]},
        ]
    ),
    "tmp.parquet",
)

array = from_parquet("tmp.parquet")

# try:
#     array.layout.field("x").array
# except AlgebraicEffect as effect:
#     print("AlgebraicEffect")
#     effect()
