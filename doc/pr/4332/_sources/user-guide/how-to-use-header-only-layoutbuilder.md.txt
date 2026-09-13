---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: ROOT C++
  language: c++
  name: root
---

How to use the header-only LayoutBuilder in C++
===============================================

```{code-cell}
:tags: [hide-cell]

// Make Awkward headers available in this notebook, because we know these headers are available from the Python sources
// Don't refer to the Git repo location, because they do not exist in sdist
#pragma cling add_include_path("../../src/awkward/_connect/header-only")
```

What is header-only Layout Builder?
-----------------------------------

The header-only Layout Builder consists of a set of compile-time, templated, static C++ classes, implemented entirely in header file which can be dropped into any external project, and easily separable from the rest of the Awkward C++ codebase.

The Layout Builder namespace consists of [14 types of Layout Builders](../_static/doxygen/LayoutBuilder_8h.html).

All Builders except `Numpy` and `Empty` can take any other Builder as template parameters.
These Builders are sufficient to build every type of Awkward Array.

Why header-only Layout Builder?
-------------------------------

The users can directly include `LayoutBuilder.h` in their compilation, rather than linking against platform-specific libraries or worrying about native dependencies. This makes the integration of Awkward Arrays into other projects easier and more portable.

The code is minimal; it does not include all of the code needed to use Awkward Arrays in Python, nor does it have helper methods to pass the data through pybind11, so that different projects can use different binding generators. The C++ users can use it to make arrays and then copy them to Python without any specialised data types - only raw buffers, strings, and integers.

How to use Layout Builders?
-----------------------------

:::{note}
A set of example projects that use the header-only layout-builder can be found [in Awkward Array's repository](https://github.com/scikit-hep/awkward/tree/main/header-only/examples).
:::

If you are using the CMake project generator, then the `awkward-headers` library can be installed using `FetchContent` for a particular version:
```cmake
include(FetchContent)

set(AWKWARD_VERSION "v2.1.0")

FetchContent_Declare(
  awkward-headers
  URL      https://github.com/scikit-hep/awkward/releases/download/${AWKWARD_VERSION}/header-only.zip
)
# Instead of using `FetchContent_MakeAvailable(awkward-headers)`, we manually load the target so
# that we can EXCLUDE_FROM_ALL
FetchContent_GetProperties(awkward-headers)
if(NOT awkward-headers_POPULATED)
  FetchContent_Populate(awkward-headers)
  add_subdirectory(${awkward-headers_SOURCE_DIR} ${awkward-headers_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
```

The loaded targets can then be linked against, e.g. to link `my_application` against the `layout-builder` target:
```cmake
target_link_libraries(my_application awkward::layout-builder)
```

If you are using a different generator, it is recommended to download these headers from the [release artifacts on GitHub](https://github.com/scikit-hep/awkward/releases). Each of the targets enumerated in `CMakeLists.txt` should be added to the include path that is passed to the compiler.


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

Second Method: The user-defined `fields_map` can be passed a parameter in `set_fields()`.

```{code-cell}
builder.set_fields(fields_map);
```

The `fields()` method can be used to check if field names are set correctly in the Record Builder or not.

```{code-cell}
std::vector<std::string> fields {"one", "two"};

auto names = builder.fields();
names
```

Assign each field content to a `fieldname_builder` builder which will be used to fill the Builder buffers.

```{code-cell}
auto& one_builder = builder.content<Field::one>();
auto& two_builder = builder.content<Field::two>();
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

We want NumPy to own the array buffers, so that they get deleted when the Awkward Array goes out of Python scope, not when the LayoutBuilder goes out of C++ scope. For the hand-off, one can allocate memory for those buffers in Python, presumably with `np.empty(nbytes, dtype=np.uint8)` and get `void*` pointers to these buffers by casting the output of `numpy_array.ctypes.data` (pointer as integer). Then we can pass everything over the border from C++ to Python using e.g. `pybind11`'s `py::buffer_protocol` for the buffers.  

Alternatively, the Python _capsule_ system can be used to tie the lifetime of the allocated buffers to the calling Python scope. `pybind11` makes this fairly trivial, and also permits us to invoke Python code _from_ C++. We can use this approach to call `ak.from_buffers` in order to build an `ak.Array`:
```cpp
template <typename T>
py::object snapshot_builder(const T& builder)
{
    // How much memory to allocate?
    std::map<std::string, size_t> names_nbytes = {};
    builder.buffer_nbytes(names_nbytes);

    // Allocate memory
    std::map<std::string, void*> buffers = {};
    for(auto it : names_nbytes) {
        uint8_t* ptr = new uint8_t[it.second];
        buffers[it.first] = (void*)ptr;
    }

    // Write non-contiguous contents to memory
    builder.to_buffers(buffers);
    auto from_buffers = py::module::import("awkward").attr("from_buffers");

    // Build Python dictionary containing arrays
    // dtypes not important here as long as they match the underlying buffer
    // as Awkward Array calls `frombuffer` to convert to the correct type
    py::dict container;
    for (auto it: buffers) {

        // Create capsule that frees the allocated data when out of scope
        py::capsule free_when_done(it.second, [](void *data) {
            uint8_t* dataPtr = reinterpret_cast<uint8_t*>(data);
            delete[] dataPtr;
        });

        // Adopt the memory filled by `to_buffers` as a NumPy array
        // We only need to return a "buffer" here, but py::array_t let's
        // us associate a capsule for destruction, which means that
        // Python can own this memory. Therefore, we use py::array_t
        uint8_t* data = reinterpret_cast<uint8_t*>(it.second);
        container[py::str(it.first)] = py::array_t<uint8_t>(
            {names_nbytes[it.first]},
            {sizeof(uint8_t)},
            data,
            free_when_done
        );
    }
    return from_buffers(builder.form(), builder.length(), container);

}
```


More Examples
-------------
Examples for other LayoutBuilders can be found [here](https://github.com/scikit-hep/awkward/blob/main/header-only/tests/test_1494-layout-builder.cpp).
