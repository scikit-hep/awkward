# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""
Pyarrow extension classes: AwkwardArrowArray and AwkwardArrowType
See: https://arrow.apache.org/docs/python/extending_types.html
"""

from __future__ import annotations

import json

import pyarrow

from awkward._nplikes.numpy_like import NumpyMetadata

np = NumpyMetadata.instance()


class AwkwardArrowArray(pyarrow.ExtensionArray):
    def to_pylist(self):
        out = super().to_pylist()
        if (
            isinstance(self.type, AwkwardArrowType)
            and self.type.node_type == "RecordArray"
            and self.type.record_is_tuple is True
        ):
            for i, x in enumerate(out):
                if x is not None:
                    out[i] = tuple(x[str(j)] for j in range(len(x)))
        return out


class AwkwardArrowType(pyarrow.ExtensionType):
    def __init__(
        self,
        storage_type,
        mask_type,
        node_type,
        mask_parameters,
        node_parameters,
        record_is_tuple,
        record_is_scalar,
        is_nonnullable_nulltype=False,
    ):
        self._mask_type = mask_type
        self._node_type = node_type
        self._mask_parameters = mask_parameters
        self._node_parameters = node_parameters
        self._record_is_tuple = record_is_tuple
        self._record_is_scalar = record_is_scalar
        self._is_nonnullable_nulltype = is_nonnullable_nulltype
        super().__init__(storage_type, "awkward")

    def __str__(self):
        return "ak:" + str(self.storage_type)

    def __repr__(self):
        return f"awkward<{self.storage_type!r}>"

    @property
    def mask_type(self):
        return self._mask_type

    @property
    def node_type(self):
        return self._node_type

    @property
    def mask_parameters(self):
        return self._mask_parameters

    @property
    def node_parameters(self):
        return self._node_parameters

    @property
    def record_is_tuple(self):
        return self._record_is_tuple

    @property
    def record_is_scalar(self):
        return self._record_is_scalar

    def __arrow_ext_class__(self):
        return AwkwardArrowArray

    def __arrow_ext_serialize__(self):
        return json.dumps(self._metadata_as_dict()).encode(errors="surrogatescape")

    def _metadata_as_dict(self):
        return {
            "mask_type": self._mask_type,
            "node_type": self._node_type,
            "mask_parameters": self._mask_parameters,
            "node_parameters": self._node_parameters,
            "record_is_tuple": self._record_is_tuple,
            "record_is_scalar": self._record_is_scalar,
            "is_nonnullable_nulltype": self._is_nonnullable_nulltype,
        }

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        # pyarrow calls this internally
        metadata = json.loads(serialized.decode(errors="surrogatescape"))
        return cls._from_metadata_object(storage_type, metadata)

    @classmethod
    def _from_metadata_object(cls, storage_type, metadata):
        return cls(
            storage_type,
            metadata["mask_type"],
            metadata["node_type"],
            metadata["mask_parameters"],
            metadata["node_parameters"],
            metadata["record_is_tuple"],
            metadata["record_is_scalar"],
            is_nonnullable_nulltype=metadata.get("is_nonnullable_nulltype", False),
        )

    @property
    def num_buffers(self):
        return self.storage_type.num_buffers

    @property
    def num_fields(self):
        return self.storage_type.num_fields

    def field(self, i: int | str):
        return self.storage_type.field(i)


pyarrow.register_extension_type(
    AwkwardArrowType(pyarrow.null(), None, None, None, None, None, None)
)


def to_awkwardarrow_storage_types(arrowtype):
    if isinstance(arrowtype, AwkwardArrowType):
        return arrowtype, arrowtype.storage_type
    else:
        return None, arrowtype
