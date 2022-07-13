# API Reference

% Build toctree from generated summaries
<!-- :::{toctree} -->
<!-- :maxdepth: 2 -->
<!-- :hidden: -->
<!-- :glob: -->
<!--  -->
<!-- _auto/* -->
<!-- ::: -->

## High-Level data types
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.Array
    awkward._v2.Record
:::

## Append-only data type
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.ArrayBuilder
:::

## Adding methods, overloading operator
...

## Describing an array
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.is_valid
    awkward._v2.validity_error
    awkward._v2.fields
    awkward._v2.type
    awkward._v2.parameters
:::

## Converting from other formats
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.from_numpy
    awkward._v2.from_iter
    awkward._v2.from_json
:::


## Converting to other formats
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.to_numpy
    awkward._v2.to_iter
    awkward._v2.to_json
:::


## Conversion functions used internally
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.to_layout
    awkward._v2.regularize_numpyarray
:::


## Alternative to filtering
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.mask
:::


## Number of elements in each list
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.num
:::


## Making and breaking arrays of records
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.zip
    awkward._v2.unzip
:::


## Manipulating records
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.with_name
    awkward._v2.with_field
:::


## Manipulating parameters
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.with_parameter
    awkward._v2.without_parameters
:::


## Broadcasting
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.broadcast_arrays
:::


## Merging arrays
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.concatenate
    awkward._v2.where
:::

## Flattening lists and missing values
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.flatten
    awkward._v2.ravel
:::

## Inserting, replacing, and checking for missing values
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.fill_none
    awkward._v2.is_none
    awkward._v2.pad_none
:::

## Converting missing values to and from empty lists 
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.firsts
    awkward._v2.singletons
:::

## Combinatorics 
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.argcartesian
    awkward._v2.argcombinations
    awkward._v2.cartesian
    awkward._v2.combinations
:::

## NumPy compatibility 
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.size
    awkward._v2.atleast_1d
    awkward._v2.cartesian
    awkward._v2.combinations
:::

## Reducers
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.all
    awkward._v2.any
    awkward._v2.argmax
    awkward._v2.argmin
    awkward._v2.count
    awkward._v2.count_nonzero
    awkward._v2.max
    awkward._v2.min
    awkward._v2.prod
    awkward._v2.sum
:::

## Non-reducers

:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.moment
    awkward._v2.mean
    awkward._v2.var
    awkward._v2.std
    awkward._v2.covar
    awkward._v2.corr
    awkward._v2.linear_fit
    awkward._v2.softmax
:::

## String behaviors
Defined in the `awkward.behaviors.string` submodule; rarely needed for analysis (strings are a built-in behavior).

## Third-party compatability

:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.numba.register
    awkward._v2.to_pandas
    awkward._v2.numexpr.evaluate
    awkward._v2.numexpr.re_evaluate
    awkward._v2.autograd.elementwise_grad
:::

## Layout nodes 
:::{eval-rst}
.. autosummary::
    :toctree: _auto
    
    awkward._v2.contents.Content
    awkward._v2.contents.EmptyArray
    awkward._v2.contents.NumpyArray
    awkward._v2.contents.RegularArray
    awkward._v2.contents.ListArray
    awkward._v2.contents.ListOffsetArray
    awkward._v2.contents.RecordArray
    awkward._v2.record.Record
    awkward._v2.contents.IndexedArray
    awkward._v2.contents.IndexedOptionArray
    awkward._v2.contents.ByteMaskedArray
    awkward._v2.contents.BitMaskedArray
    awkward._v2.contents.UnmaskedArray
    awkward._v2.contents.UnionArray
:::

## Layout-level ArrayBuilder
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.contents.ArrayBuilder
:::

## Index for layout nodes
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.contents.Index
:::

## Identities for layout nodes
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.contents.Identities
:::

## High-level data types
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.types.Type
    awkward._v2.types.ArrayType
    awkward._v2.types.UnknownType
    awkward._v2.types.PrimitiveType
    awkward._v2.types.RegularType
    awkward._v2.types.ListType
    awkward._v2.types.RecordType
    awkward._v2.types.OptionType
    awkward._v2.types.UnionType
:::


## Low-level array forms
:::{eval-rst}
.. autosummary::
    :toctree: _auto

    awkward._v2.forms.Form
    awkward._v2.forms.EmptyForm
    awkward._v2.forms.NumpyForm
    awkward._v2.forms.RegularForm
    awkward._v2.forms.ListForm
    awkward._v2.forms.ListOffsetForm
    awkward._v2.forms.RecordForm
    awkward._v2.forms.IndexedForm
    awkward._v2.forms.IndexedOptionForm
    awkward._v2.forms.ByteMaskedForm
    awkward._v2.forms.BitMaskedForm
    awkward._v2.forms.UnmaskedForm
    awkward._v2.forms.UnionForm
:::

## Internal implementation

The rest of the classes and functions described here are not part of the public interface. Either the objects or the submodules begin with an underscore, indicating that they can freely change from one version to the next.

## More documentation

The Awkward Array project is divided into 3 layers with 5 main components.

```{image} _static/awkward-1-0-layers.svg
:width: 500px
:align: center
```

The C++ classes, cpu-kernels, and gpu-kernels are described in the [C++ API reference](_static/index.html>).

The kernels (cpu-kernels and cuda-kernels) are documented on the {doc}`_auto/kernels` page, with interfaces and normative Python implementations.

