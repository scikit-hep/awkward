# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class ParquetReader:
    @classmethod
    def _create(cls, name, path, fs, columns, options):
        parquet_dataset = ak._v2._connect.pyarrow.import_pyarrow_dataset(name)
        dataset = parquet_dataset.dataset(path, format="parquet", filesystem=fs)
        form = ak._v2._connect.pyarrow.form_handle_arrow(
            dataset.schema, pass_empty_field=True
        )
        if columns is None:
            columns = form.columns()
        else:
            form, columns = form.select_columns(columns)
        return ParquetReader(dataset, form, columns, options)

    def __init__(self, dataset, form, columns, options):
        self._dataset = dataset
        self._form = form
        self._columns = columns
        self._options = options

    @property
    def form(self):
        return self._form

    @property
    def columns(self):
        return self._columns

    def __len__(self):
        return self._dataset.count_rows()

    @property
    def type(self):
        return ak._v2.types.ArrayType(
            self._form.type_from_behavior(self._options["behavior"]), len(self)
        )

    def __repr__(self):
        length = 80 - len(type(self).__name__) - len("< type=''>")
        typestr = repr(str(self.type))[1:-1]
        if len(typestr) > length:
            typestr = "'" + typestr[: length - 3] + "...'"
        else:
            typestr = "'" + typestr + "'"
        return f"<{type(self).__name__} type={typestr}>"

    def select(self, columns):
        form, columns = self._form.select_columns(columns)
        return ParquetReader(self._dataset, form, columns, self._options)

    def read(self):
        if self._options["list_indicator"] is None:
            columns = self._columns
        else:
            columns = self._form.columns(self._options["list_indicator"])

        table = self._dataset.to_table(columns=columns)
        return ak._v2.operations.convert.ak_from_arrow._impl(
            table,
            self._options["conservative_optiontype"],
            self._options["highlevel"],
            self._options["behavior"],
        )


def from_parquet(
    path,
    storage_options=None,
    columns=None,
    as_reader=False,
    conservative_optiontype=False,
    list_indicator="item",
    highlevel=True,
    behavior=None,
):
    import awkward._v2._connect.pyarrow  # noqa: F401

    options = {
        "conservative_optiontype": conservative_optiontype,
        "list_indicator": list_indicator,
        "highlevel": highlevel,
        "behavior": behavior,
    }

    fsspec = ak._v2._connect.pyarrow.import_fsspec("ak._v2.from_parquet")

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )

    if len(paths) == 1:
        if fs.isfile("/".join([paths[0], "_metadata"])) or fs.isfile(paths[0]):
            reader = ParquetReader._create(
                "ak._v2.from_parquet", paths[0], fs, columns, options
            )
        else:
            reader = ParquetReader._create(
                "ak._v2.from_parquet", fs.find(paths[0]), fs, columns, options
            )
    else:
        reader = ParquetReader._create(
            "ak._v2.from_parquet", paths, fs, columns, options
        )

    if as_reader:
        return reader
    else:
        return reader.read()
