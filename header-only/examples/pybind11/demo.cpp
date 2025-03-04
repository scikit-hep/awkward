// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "awkward/LayoutBuilder.h"

namespace py = pybind11;

// Defines taken from how-to-use-header-only-layoutbuilder.md
using UserDefinedMap = std::map<std::size_t, std::string>;
template<class... BUILDERS>
using RecordBuilder = awkward::LayoutBuilder::Record<UserDefinedMap, BUILDERS...>;
template<std::size_t field_name, class BUILDER>
using RecordField = awkward::LayoutBuilder::Field<field_name, BUILDER>;
template<class PRIMITIVE, class BUILDER>
using ListOffsetBuilder = awkward::LayoutBuilder::ListOffset<PRIMITIVE, BUILDER>;
template<class PRIMITIVE>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<PRIMITIVE>;
enum Field : std::size_t {
    one, two
};
using MyBuilder = RecordBuilder<
        RecordField<Field::one, NumpyBuilder<double>>,
        RecordField<Field::two, ListOffsetBuilder<int64_t,
                NumpyBuilder<int32_t>>>
>;


/**
 * Create a snapshot of the given builder, and return an `ak.Array` pyobject
 * @tparam T type of builder
 * @param builder builder
 * @return pyobject of Awkward Array
 */
template<typename T>
py::object snapshot_builder(const T &builder) {
    // We need NumPy (to allocate arrays) and Awkward Array (ak.from_buffers).
    // pybind11 will raise a ModuleNotFoundError if they aren't installed.
    auto np = py::module::import("numpy");
    auto ak = py::module::import("awkward");

    auto dtype_u1 = np.attr("dtype")("u1");

    // How much memory to allocate?
    std::map<std::string, size_t> names_nbytes;
    builder.buffer_nbytes(names_nbytes);

    // Ask NumPy to allocate memory and get pointers to the raw buffers.
    py::dict py_container;
    std::map<std::string, void*> cpp_container;
    for (auto name_nbytes : names_nbytes) {
      py::object array = np.attr("empty")(name_nbytes.second, dtype_u1);

      size_t pointer = py::cast<size_t>(array.attr("ctypes").attr("data"));
      void* raw_data = (void*)pointer;

      py::str py_name(name_nbytes.first);
      py_container[py_name] = array;
      cpp_container[name_nbytes.first] = raw_data;
    }

    // Write non-contiguous contents to memory.
    builder.to_buffers(cpp_container);

    // Build Python dictionary containing arrays.
    return ak.attr("from_buffers")(builder.form(), builder.length(), py_container);
}


/**
 * Create demo array, and return its snapshot
 * @return pyobject of Awkward Array
 */
py::object create_demo_array() {
    UserDefinedMap fields_map({
        {Field::one, "one"},
        {Field::two, "two"}
    });

    RecordBuilder<
            RecordField<Field::one, NumpyBuilder<double>>,
            RecordField<Field::two, ListOffsetBuilder<int64_t,
                    NumpyBuilder<int32_t>>>
    > builder(fields_map);

    auto &one_builder = builder.content<Field::one>();
    auto &two_builder = builder.content<Field::two>();

    one_builder.append(1.1);

    auto &two_subbuilder = two_builder.begin_list();
    two_subbuilder.append(1);
    two_builder.end_list();

    one_builder.append(2.2);

    two_builder.begin_list();
    two_subbuilder.append(1);
    two_subbuilder.append(2);
    two_builder.end_list();

    one_builder.append(3.3);
    size_t data_size = 3;
    int32_t data[3] = {1, 2, 3};

    two_builder.begin_list();
    two_subbuilder.extend(data, data_size);
    two_builder.end_list();

    return snapshot_builder(builder);
}


PYBIND11_MODULE(demo, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("create_demo_array", &create_demo_array, "A function that creates an awkward array");
}
