# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Mapping

import awkward as ak


class ParquetReader:
    @classmethod
    def _create(cls, name, path, fs, columns, behavior):
        parquet_dataset = ak._v2._connect.pyarrow.import_pyarrow_dataset(name)
        dataset = parquet_dataset.dataset(path, format="parquet", filesystem=fs)
        form = ak._v2._connect.pyarrow.form_handle_arrow(
            dataset.schema, pass_empty_field=True
        )
        if columns is None:
            columns = form.columns()
        else:
            form, columns = form.select_columns(columns)
        return ParquetReader(dataset, form, columns, behavior)

    def __init__(self, dataset, form, columns, behavior):
        self._dataset = dataset
        self._form = form
        self._columns = columns
        self.behavior = behavior

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, value):
        if value is None or isinstance(value, Mapping):
            self._behavior = value
        else:
            raise ak._v2._util.error(TypeError("behavior must be None or a dict"))

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
            self._form.type_from_behavior(self._behavior), len(self)
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
        return ParquetReader(self._dataset, form, columns, self._behavior)


def from_parquet(path, storage_options=None, columns=None, behavior=None):
    import awkward._v2._connect.pyarrow  # noqa: F401

    fsspec = ak._v2._connect.pyarrow.import_fsspec("ak._v2.from_parquet")

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )

    if len(paths) == 1:
        if fs.isfile("/".join([paths[0], "_metadata"])) or fs.isfile(paths[0]):
            return ParquetReader._create(
                "ak._v2.from_parquet", paths[0], fs, columns, behavior
            )
        else:
            return ParquetReader._create(
                "ak._v2.from_parquet", fs.find(paths[0]), fs, columns, behavior
            )
    else:
        return ParquetReader._create(
            "ak._v2.from_parquet", paths, fs, columns, behavior
        )
