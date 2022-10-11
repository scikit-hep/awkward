.. toctree::
    :caption: High-level data types

    generated/ak.Array.rst
    generated/ak.Record.rst

.. toctree::
    :caption: Append-only data type

    generated/ak.ArrayBuilder.rst

.. toctree::
    :caption: Describing an array

    generated/ak.fields.rst
    generated/ak.is_valid.rst
    generated/ak.parameters.rst
    generated/ak.is_tuple.rst
    generated/ak.type.rst
    generated/ak.validity_error.rst

.. toctree::
    :caption: Converting from other formats

    generated/ak.from_arrow.rst
    generated/ak.from_arrow_schema.rst
    generated/ak.from_buffers.rst
    generated/ak.from_cupy.rst
    generated/ak.from_iter.rst
    generated/ak.from_jax.rst
    generated/ak.from_json.rst
    generated/ak.from_numpy.rst
    generated/ak.from_parquet.rst
    generated/ak.from_rdataframe.rst
    generated/ak.from_avro_file.rst

.. toctree::
    :caption: Converting to other formats

    generated/ak.to_arrow.rst
    generated/ak.to_arrow_table.rst
    generated/ak.to_buffers.rst
    generated/ak.to_cupy.rst
    generated/ak.to_dataframe.rst
    generated/ak.to_jax.rst
    generated/ak.to_json.rst
    generated/ak.to_list.rst
    generated/ak.to_numpy.rst
    generated/ak.to_parquet.rst
    generated/ak.to_rdataframe.rst

.. toctree::
    :caption: Number of elements in each list

    generated/ak.count.rst
    generated/ak.count_nonzero.rst
    generated/ak.num.rst

.. toctree::
    :caption: Making and breaking arrays of records

    generated/ak.unzip.rst
    generated/ak.zip.rst

.. toctree::
    :caption: Manipulating records

    generated/ak.with_field.rst
    generated/ak.with_name.rst

.. toctree::
    :caption: Manipulating parameters

    generated/ak.with_parameter.rst
    generated/ak.without_parameters.rst

.. toctree::
    :caption: Broadcasting

    generated/ak.broadcast_arrays.rst

.. toctree::
    :caption: Merging arrays

    generated/ak.concatenate.rst
    generated/ak.where.rst

.. toctree::
    :caption: Flattening lists and missing values

    generated/ak.flatten.rst
    generated/ak.ravel.rst
    generated/ak.unflatten.rst

.. toctree::
    :caption: Working with missing values

    generated/ak.mask.rst
    generated/ak.is_none.rst
    generated/ak.fill_none.rst
    generated/ak.pad_none.rst
    generated/ak.firsts.rst
    generated/ak.singletons.rst

.. toctree::
    :caption: Combinatorics

    generated/ak.argcartesian.rst
    generated/ak.argcombinations.rst
    generated/ak.cartesian.rst
    generated/ak.combinations.rst

.. toctree::
    :caption: NumPy compatibility

    generated/ak.isclose.rst
    generated/ak.ones_like.rst
    generated/ak.zeros_like.rst
    generated/ak.full_like.rst

.. toctree::
    :caption: Reducers

    generated/ak.argsort.rst
    generated/ak.sort.rst
    generated/ak.all.rst
    generated/ak.any.rst
    generated/ak.argmax.rst
    generated/ak.argmin.rst
    generated/ak.max.rst
    generated/ak.min.rst
    generated/ak.nanargmax.rst
    generated/ak.nanargmin.rst
    generated/ak.nanmax.rst
    generated/ak.nanmin.rst
    generated/ak.nanprod.rst
    generated/ak.nansum.rst
    generated/ak.prod.rst
    generated/ak.sum.rst

.. toctree::
    :caption: Non-reducers

    generated/ak.corr.rst
    generated/ak.covar.rst
    generated/ak.linear_fit.rst
    generated/ak.mean.rst
    generated/ak.moment.rst
    generated/ak.nanmean.rst
    generated/ak.nanstd.rst
    generated/ak.nanvar.rst
    generated/ak.ptp.rst
    generated/ak.softmax.rst
    generated/ak.std.rst
    generated/ak.var.rst

.. toctree::
    :caption: Working with categorical arrays

    generated/ak.categories.rst
    generated/ak.is_categorical.rst
    generated/ak.to_categorical.rst
    generated/ak.from_categorical.rst

.. toctree::
    :caption: Converting between regular and ragged arrays

    generated/ak.to_regular.rst
    generated/ak.from_regular.rst

.. toctree::
    :caption: Value and type conversions

    generated/ak.strings_astype.rst
    generated/ak.values_astype.rst
    generated/ak.nan_to_none.rst
    generated/ak.nan_to_num.rst

.. toctree::
    :caption: Low-level layouts

    generated/ak.to_layout.rst
    generated/ak.contents.BitMaskedArray.rst
    generated/ak.contents.ByteMaskedArray.rst
    generated/ak.contents.Content.rst
    generated/ak.contents.EmptyArray.rst
    generated/ak.contents.IndexedArray.rst
    generated/ak.contents.IndexedOptionArray.rst
    generated/ak.contents.ListArray.rst
    generated/ak.contents.ListOffsetArray.rst
    generated/ak.contents.NumpyArray.rst
    generated/ak.contents.RecordArray.rst
    generated/ak.contents.RegularArray.rst
    generated/ak.contents.UnionArray.rst
    generated/ak.contents.UnmaskedArray.rst
    generated/ak.record.Record.rst

.. toctree::
    :caption: Index for layout nodes

    generated/ak.index.Index.rst
    generated/ak.index.Index32.rst
    generated/ak.index.Index64.rst
    generated/ak.index.Index8.rst
    generated/ak.index.IndexU32.rst
    generated/ak.index.IndexU8.

.. toctree::
    :caption: Identities for layout nodes

    generated/ak.identifier.Identifier.rst

.. toctree::
    :caption: String behaviors

    generated/ak.ByteBehavior.rst
    generated/ak.ByteStringBehavior.rst
    generated/ak.CategoricalBehavior.rst
    generated/ak.CharBehavior.rst
    generated/ak.StringBehavior.rst

.. toctree::
    :caption: Converting between backends

    generated/ak.backend.rst
    generated/ak.to_backend.rst

.. toctree::
    :caption: Behavior classes

    generated/ak.mixin_class.rst
    generated/ak.mixin_class_method.rst

.. toctree::
    :caption: Indexing and grouping

    generated/ak.local_index.rst
    generated/ak.run_lengths.rst

.. toctree::
    :caption: Copying and packing arrays
    generated/ak.packed.rst
    generated/ak.copy.rst

.. toctree::
    :caption: Layout node transformations
    generated/ak.transform.rst


.. toctree::
    :caption: High-level data types

    generated/ak.types.ArrayType.rst
    generated/ak.types.ListType.rst
    generated/ak.types.NumpyType.rst
    generated/ak.types.OptionType.rst
    generated/ak.types.RecordType.rst
    generated/ak.types.RegularType.rst
    generated/ak.types.Type.rst
    generated/ak.types.UnionType.rst
    generated/ak.types.UnknownType.rst
    generated/ak.types.dtype_to_primitive.rst
    generated/ak.types.from_datashape.rst
    generated/ak.types.is_primitive.rst
    generated/ak.types.primitive_to_dtype.rst

.. toctree::
    :caption: Low-level array forms

    generated/ak.forms.from_dict.rst
    generated/ak.forms.from_json.rst
    generated/ak.forms.BitMaskedForm.rst
    generated/ak.forms.ByteMaskedForm.rst
    generated/ak.forms.EmptyForm.rst
    generated/ak.forms.Form.rst
    generated/ak.forms.IndexedForm.rst
    generated/ak.forms.IndexedOptionForm.rst
    generated/ak.forms.ListForm.rst
    generated/ak.forms.ListOffsetForm.rst
    generated/ak.forms.NumpyForm.rst
    generated/ak.forms.RecordForm.rst
    generated/ak.forms.RegularForm.rst
    generated/ak.forms.UnionForm.rst
    generated/ak.forms.UnmaskedForm.rst

.. toctree::
    :caption: Third-party integration

    generated/ak.numba.register_and_check.rst
    generated/ak.jax.register_and_check.rst
    generated/ak.jax.assert_registered.rst
    generated/ak.jax.import_jax.rst

.. toctree::
    :caption: Forth virtual machine

    awkwardforth.rst