# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import ctypes
import os

import cppyy
import ROOT

import awkward as ak
import awkward._connect.cling
import awkward._lookup
from awkward.types.numpytype import primitive_to_dtype

cpp_type_of = {
    "bool": "bool",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
    "complex64": "std::complex<float>",
    "complex128": "std::complex<double>",
    "datetime64": "std::time_t",
    "timedelta64": "std::difftime",
}

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


cppyy.add_include_path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "include",
        )
    )
)
cppyy.add_include_path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.pardir,
            "header-only",
        )
    )
)
compiler = ROOT.gInterpreter.Declare


done = compiler(
    """
#include "rdataframe/jagged_builders.h"
"""
)
assert done is True


def from_rdataframe(data_frame, columns):
    def form_dtype(form):
        if isinstance(form, ak.forms.NumpyForm) and form.inner_shape == ():
            return primitive_to_dtype(form.primitive)
        elif isinstance(form, ak.forms.ListOffsetForm):
            return form_dtype(form.content)

    def empty_buffers(cpp_buffers_self, names_nbytes):
        buffers = {}
        for item in names_nbytes:
            buffers[item.first] = ak._nplikes.numpy.empty(item.second)
            cpp_buffers_self.append(
                item.first,
                buffers[item.first].ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            )
        return buffers

    def cpp_builder_type(depth, data_type):
        if depth == 1:
            return f"awkward::LayoutBuilder::Numpy<{data_type}>>"
        else:
            return (
                "awkward::LayoutBuilder::ListOffset<int64_t, "
                + cpp_builder_type(depth - 1, data_type)
                + ">"
            )

    def cpp_fill_offsets_and_flatten(depth):
        if depth == 1:
            return "\nfor (auto it : vec1) {\n" + "  builder1.append(it);\n" + "}\n"
        else:
            return (
                f"for (auto const& vec{depth - 1} : vec{depth}) "
                + "{\n"
                + f"  auto& builder{depth - 1} = builder{depth}.begin_list();\n"
                + "  "
                + cpp_fill_offsets_and_flatten(depth - 1)
                + "\n"
                + f"  builder{depth}.end_list();\n"
                + "}\n"
            )

    def cpp_fill_function(depth):
        if depth == 1:
            return (
                "template<class BUILDER, typename PRIMITIVE>\n"
                + "void\n"
                + "fill_from(BUILDER& builder, ROOT::RDF::RResultPtr<std::vector<PRIMITIVE>>& result) {"
                + "  for (auto it : result) {\n"
                + "    builder.append(it);\n"
                + "  }\n"
                + "}\n"
            )
        else:
            return (
                "template<class BUILDER, typename PRIMITIVE>\n"
                + "void\n"
                + f"fill_offsets_and_flatten{depth}(BUILDER& builder{depth}, ROOT::RDF::RResultPtr<std::vector<PRIMITIVE>>& result) "
                + "{\n"
                + f"  for (auto const& vec{depth - 1} : result) "
                + "{\n"
                + f"  auto& builder{depth - 1} = builder{depth}.begin_list();\n"
                + "  "
                + cpp_fill_offsets_and_flatten(depth - 1)
                + "\n"
                + f"  builder{depth}.end_list();\n"
                + "}\n"
                + "}\n"
            )

    is_indexed = True if "awkward_index_" in data_frame.GetColumnNames() else False

    # Register Take action for each column
    # 'Take' is a lazy action:
    result_ptrs = {}
    column_types = {}
    contents_index = None
    columns = (
        columns + ("awkward_index_",)
        if (is_indexed and "awkward_index_" not in columns)
        else columns
    )
    for col in columns:
        column_types[col] = data_frame.GetColumnType(col)
        result_ptrs[col] = data_frame.Take[column_types[col]](col)

    contents = {}
    awkward_contents = {}
    contents_index = {}
    for col in columns:
        col_type = column_types[col]
        if ROOT.awkward.is_awkward_type[col_type]():  # Retrieve Awkward arrays

            # ROOT::RDF::RResultPtr<T>::begin Returns an iterator to the beginning of
            # the contained object if this makes sense, throw a compilation error otherwise.
            #
            # Does not trigger event loop and execution of all actions booked in
            # the associated RLoopManager.
            lookup = result_ptrs[col].begin().lookup()
            generator = lookup[col].generator
            layout = generator.tolayout(lookup[col], 0, ())
            awkward_contents[col] = layout

        else:  # Convert the C++ vectors to Awkward arrays
            form_str = ROOT.awkward.type_to_form[col_type](0)

            if form_str == "unsupported type":
                raise ak._errors.wrap_error(
                    TypeError(f'"{col}" column\'s type "{col_type}" is not supported.')
                )

            form = ak.forms.from_json(form_str)

            list_depth = form.purelist_depth
            form_dtype_name = form_dtype(form).name
            data_type = cpp_type_of[form_dtype_name]

            # pull in the CppBuffers (after which we can import from it)
            CppBuffers = cppyy.gbl.awkward.CppBuffers[col_type]
            cpp_buffers_self = CppBuffers(result_ptrs[col])

            if isinstance(form, ak.forms.NumpyForm):

                NumpyBuilder = cppyy.gbl.awkward.LayoutBuilder.Numpy[data_type]
                builder = NumpyBuilder()
                builder_type = type(builder).__cpp_name__

                cpp_buffers_self.fill_from[builder_type, col_type](
                    builder, result_ptrs[col]
                )

                names_nbytes = cpp_buffers_self.names_nbytes[builder_type](builder)
                buffers = empty_buffers(cpp_buffers_self, names_nbytes)
                cpp_buffers_self.to_char_buffers[builder_type](builder)

            elif isinstance(form, ak.forms.ListOffsetForm):
                if isinstance(form.content, ak.forms.NumpyForm):
                    # NOTE: list_depth == 2 or 1 if its the list of strings
                    list_depth = 2

                ListOffsetBuilder = cppyy.gbl.awkward.LayoutBuilder.ListOffset[
                    "int64_t",
                    cpp_builder_type(list_depth - 1, data_type),
                ]
                builder = ListOffsetBuilder()
                builder_type = type(builder).__cpp_name__

                if not hasattr(
                    cppyy.gbl.awkward, f"fill_offsets_and_flatten{list_depth}"
                ):
                    done = cppyy.cppdef(
                        "namespace awkward {" + cpp_fill_function(list_depth) + "}"
                    )
                    assert done is True

                fill_from_func = getattr(
                    cppyy.gbl.awkward, f"fill_offsets_and_flatten{list_depth}"
                )
                fill_from_func[builder_type, col_type](builder, result_ptrs[col])
            else:
                raise ak._errors.wrap_error(
                    AssertionError(f"unrecognized Form: {type(form)}")
                )

            names_nbytes = cpp_buffers_self.names_nbytes[builder_type](builder)
            buffers = empty_buffers(cpp_buffers_self, names_nbytes)
            cpp_buffers_self.to_char_buffers[builder_type](builder)

            array = ak.from_buffers(
                form,
                builder.length(),
                buffers,
            )

            if col == "awkward_index_":
                contents_index = ak.index.Index64(
                    array.layout.to_numpy(allow_missing=True)
                )
            else:
                contents[col] = array.layout

    for col, content in awkward_contents.items():
        # wrap Awkward array in IndexedArray only if needed
        if contents_index is not None and len(contents_index) < len(content):
            array = ak._util.wrap(
                ak.contents.IndexedArray(contents_index, content),
                highlevel=True,
            )
            contents[col] = array.layout
        else:
            contents[col] = content

    return ak._util.wrap(
        ak.contents.RecordArray(list(contents.values()), list(contents.keys())),
        highlevel=True,
    )
