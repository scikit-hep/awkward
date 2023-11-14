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
    // How much memory to allocate?
    std::map <std::string, size_t> names_nbytes = {};
    builder.buffer_nbytes(names_nbytes);

    // Allocate memory
    std::map<std::string, void *> buffers = {};
    for (auto it: names_nbytes) {
        uint8_t *ptr = new uint8_t[it.second];
        buffers[it.first] = (void *) ptr;
    }

    // Write non-contiguous contents to memory
    builder.to_buffers(buffers);
    auto from_buffers = py::module::import("awkward").attr("from_buffers");

    // Build Python dictionary containing arrays
    // dtypes not important here as long as they match the underlying buffer
    // as Awkward Array calls `frombuffer` to convert to the correct type
    py::dict container;
    for (auto it: buffers) {

        py::capsule free_when_done(it.second, [](void *data) {
            uint8_t *dataPtr = reinterpret_cast<uint8_t *>(data);
            delete[] dataPtr;
        });

        uint8_t *data = reinterpret_cast<uint8_t *>(it.second);
        container[py::str(it.first)] = py::array_t<uint8_t>(
                {names_nbytes[it.first]},
                {sizeof(uint8_t)},
                data,
                free_when_done
        );
    }
    return from_buffers(builder.form(), builder.length(), container);

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
