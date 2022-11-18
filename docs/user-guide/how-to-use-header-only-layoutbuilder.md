---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: ROOT C++
  language: c++
  name: root
---

How to use the header-only LayoutBuilder in C++
===============================================

```{code-cell}
:tags: [hide-cell]

// Make Awkward headers available in this notebook
#pragma cling add_include_path("../../src/awkward/_connect/header-only")
```

What is header-only Layout Builder?
-----------------------------------

The header-only Layout Builder consists of a set of compile-time, templated, static C++ classes, implemented entirely in header file which can be dropped into any external project, and easily separable from the rest of the Awkward C++ codebase.

The Layout Builder namespace consists of [14 types of Layout Builders](https://awkward-array.readthedocs.io/en/main/_static/doxygen/LayoutBuilder_8h.html).

All Builders except `Numpy` and `Empty` can take any other Builder as template parameters.
These Builders are sufficient to build every type of Awkward Array.

Why header-only Layout Builder?
-------------------------------

The users can directly include `LayoutBuilder.h` in their compilation, rather than linking against platform-specific libraries or worrying about native dependencies. This makes the integration of Awkward Arrays into other projects easier and more portable.

The code is minimal; it does not include all of the code needed to use Awkward Arrays in Python, nor does it have helper methods to pass the data through pybind11, so that different projects can use different binding generators. The C++ users can use it to make arrays and then copy them to Python without any specialised data types - only raw buffers, strings, and integers.

How to use Layout Builders?
-----------------------------

The following cpp-headers are needed to use Layout Builders to use the header-only `LayoutBuilder`.

1. BuilderOptions.h
2. GrowableBuffer.h
3. LayoutBuilder.h
4. utils.h

It is recommended to download these headers from the [release artifacts on GitHub](https://github.com/scikit-hep/awkward/releases)

Awkward Array can be installed from PyPI using pip:

```shell
pip install awkward
```

To get the `-I` compiler flags needed to pick up the LayoutBuilder from this installation:

```shell
python -m awkward.config --cflags
```

A user would need to pass these options to the compiler in order to use it.

Three phases of using Layout Builder
------------------------------------

1. *Constructing a Layout Builder:* from variadic templates. (It is implicit template instantiation).
2. *Filling the Layout Builder:* while repeatedly walking over the raw pointers within the LayoutBuilder.
3. *Taking the data out to user allocated buffers:* Then user can pass them to Python if they want.

Example
-------

Below is an example for creating RecordArray with NumpyArray and ListOffsetArray as fields.

First, include the LayoutBuilder header file. Note that only `LayoutBuilder.h` needs to be included in the example since the other header-only files required are already included in its implementation.

```{code-cell}
#include "awkward/LayoutBuilder.h"
```

The Record Builder content is a heterogeneous type container (std::tuple) which can take other Builders as template parameters. The field names are non-type template parameters defined by a user. Note, it is not possible to template on `std::string` because this feature comes only from `C++20`. That is why, a user-defined `field_map` with enumerated type field ID as keys and field names as value has to provided for passing the field names as template parameters to the `Record` Builder. If multiple `Record` Builders are used in a Builder, then a user-defined map has to be provided for each of the `Record` Builders used.

```{code-cell}
enum Field : std::size_t {one, two};

using UserDefinedMap = std::map<std::size_t, std::string>;

UserDefinedMap fields_map({
    {Field::one, "one"},
    {Field::two, "two"}
});
```

```{code-cell}
template<class... BUILDERS>
using RecordBuilder = awkward::LayoutBuilder::Record<UserDefinedMap, BUILDERS...>; 

template<std::size_t field_name, class BUILDER>
using RecordField = awkward::LayoutBuilder::Field<field_name, BUILDER>;
```

In the ListOffset Builder, there is an option to use 64-bit signed integers `int64`, 32-bit signed integers `int32` or 32-bit unsigned integers `uint32` as the type for list offsets.

Type alias can be used for each builder class.

```{code-cell}
template<class PRIMITIVE, class BUILDER>
using ListOffsetBuilder = awkward::LayoutBuilder::ListOffset<PRIMITIVE, BUILDER>;

template<class PRIMITIVE>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<PRIMITIVE>;
```

The builder is defined as demonstrated below. To set the field names, there are two methods:

First Method: The user-defined `fields_map` can be passed as a parameter in the object of the builder.

```{code-cell}
RecordBuilder<
  RecordField<Field::one, NumpyBuilder<double>>,
  RecordField<Field::two, ListOffsetBuilder<int64_t,
      NumpyBuilder<int32_t>>>
> builder(fields_map);
```

Second Method: The user-defined `fields_map` can be passed a parameter in `set_field_names()`.

```{code-cell}
builder.set_field_names(fields_map);
```

The `field_names()` method can be used to check if field names are set correctly in the Record Builder or not.

```{code-cell}
std::vector<std::string> fields {"one", "two"};

auto names = builder.field_names();
names
```

Assign each field content to a `fieldname_builder` builder which will be used to fill the Builder buffers.

```{code-cell}
auto& one_builder = builder.field<Field::one>();
auto& two_builder = builder.field<Field::two>();
```

Append the data in the fields using `append()`. In case of ListOffsetArray, append the data between `begin_list()` and `end_list()`.

```{code-cell}
one_builder.append(1.1);
auto& two_subbuilder = two_builder.begin_list();
two_subbuilder.append(1);
two_builder.end_list();

one_builder.append(2.2);
two_builder.begin_list();
two_subbuilder.append(1);
two_subbuilder.append(2);
two_builder.end_list();

one_builder.append(3.3);
```

Check the validity of the buffer by `is_valid()` to make sure there are no errors.

In this example, since the Record `node0` has field `two` length = 2 while the first field has length = 3, an error is generated.

```{code-cell}
std::string error;
builder.is_valid(error)
```

We can inspect the error message:

```{code-cell}
error
```

If you need to append an entire array in one go, `extend()` can be used which takes pointer to the array and the size of array as paramaters. Note that it is just an interface and not actually faster than calling append many times.

```{code-cell}
size_t data_size = 3;
int32_t data[3] = {1, 2, 3};

two_builder.begin_list();
two_subbuilder.extend(data, data_size);
two_builder.end_list();
```

Now, the length of all fields in Record `node0` are equal, no error is generated.

The `is_valid()` method can be called on every entry if you want to trade safety for speed.

```{code-cell}
builder.is_valid(error)
```

Retrieve the information needed to allocate the empty buffers as a map of their names (the form node keys as defined by the LayoutBuilder) to their sizes (in bytes).

```{code-cell}
std::map<std::string, size_t> names_nbytes = {};
builder.buffer_nbytes(names_nbytes);
names_nbytes
```

Next, allocate the memory for these buffers using the user-given pointers and the same names/sizes as above. Then, let the LayoutBuilder fill these buffers with `to_buffers()` method.

```{code-cell}
std::map<std::string, void*> buffers = {};
for(auto it : names_nbytes) {
    uint8_t* ptr = new uint8_t[it.second];
    buffers[it.first] = (void*)ptr;
}

builder.to_buffers(buffers);
```

Now, let's look at the _form_ of the builder. A Form is a unique description of an Awkward Array and returns a JSON-like `std::string` and its form keys. The is the Awkward Form generated for this example.

```{code-cell}
std::cout << builder.form() << std::endl;
```

Passing from C++ to Python
--------------------------

We want NumPy to own the array buffers, so that they get deleted when the Awkward Array goes out of Python scope, not when the LayoutBuilder goes out of C++ scope. For the hand-off, you can allocate memory for those buffers in Python, presumably with `np.empty(nbytes, dtype=np.uint8)` and get void* pointers to these buffers by casting the output of `numpy_array.ctypes.data` (pointer as integer).

Now you can pass everything over the border from C++ to Python using pybind11's `py::buffer_protocol` for the buffers, as well as an integer for the length and a string for the Form.

More Examples
-------------
The examples for other Layout Builders can be found [here](https://github.com/scikit-hep/awkward/blob/main/tests-cpp/test_1494-layout-builder.cpp).
