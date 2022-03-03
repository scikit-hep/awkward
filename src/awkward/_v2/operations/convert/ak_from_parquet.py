# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Mapping

import awkward as ak


class ParquetDataset:
    @classmethod
    def _create(self, name, path, fs, behavior):
        parquet_dataset = ak._v2._connect.pyarrow.import_pyarrow_dataset(name)
        return ParquetDataset(
            parquet_dataset.dataset(path, format="parquet", filesystem=fs),
            None,
            behavior,
        )

    def __init__(self, dataset, columns, behavior):
        self._dataset = dataset
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
    def schema(self):
        return self._dataset.schema

    @property
    def form(self):
        return ak._v2._connect.pyarrow.form_handle_arrow(
            self._dataset.schema, pass_empty_field=True
        )

    def __len__(self):
        return self._dataset.count_rows()

    @property
    def type(self):
        return ak._v2.types.ArrayType(
            self.form.type_from_behavior(self._behavior), len(self)
        )

    def __repr__(self):
        typestr = repr(str(self.type))[1:-1]
        if len(typestr) > 80 - 24:
            typestr = "'" + typestr[: 80 - 24 - 3] + "...'"
        else:
            typestr = "'" + typestr + "'"
        return f"<ParquetDataset type={typestr}>"


def from_parquet(path, storage_options=None, behavior=None):
    import awkward._v2._connect.pyarrow  # noqa: F401

    fsspec = ak._v2._connect.pyarrow.import_fsspec("ak._v2.from_parquet")

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )

    if len(paths) == 1:
        if fs.isfile("/".join([paths[0], "_metadata"])) or fs.isfile(paths[0]):
            return ParquetDataset._create("ak._v2.from_parquet", paths[0], fs, behavior)
        else:
            return ParquetDataset._create(
                "ak._v2.from_parquet", fs.find(paths[0]), fs, behavior
            )
    else:
        return ParquetDataset._create("ak._v2.from_parquet", paths, fs, behavior)
