# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy
import ctypes
import os
from awkward._v2.types.numpytype import primitive_to_dtype

cpp_type_of = {
    "float64": "double",
    "int64": "int64_t",
    "complex128": "std::complex<double>",
    "uint8": "uint8_t",
}

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


cppyy.add_include_path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir,
            os.pardir,
            "cpp-headers",
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
    def supported(form):
        if form.purelist_depth == 1:
            # special case for a list of strings form
            return isinstance(
                form, (ak._v2.forms.ListOffsetForm, ak._v2.forms.NumpyForm)
            )
        else:
            return isinstance(form, ak._v2.forms.ListOffsetForm) and supported(
                form.content
            )

    def form_dtype(form):
        if form.purelist_depth == 1:
            # special case for a list of strings form
            return (
                primitive_to_dtype(form.content.primitive)
                if isinstance(form, ak._v2.forms.ListOffsetForm)
                else primitive_to_dtype(form.primitive)
            )
        else:
            return form_dtype(form.content)

    def empty_buffers(cpp_buffers_self, names_nbytes):
        buffers = {}
        for item in names_nbytes:
            buffers[item.first] = ak.nplike.numpy.empty(item.second)
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

    # Register Take action for each column
    # 'Take' is a lazy action:
    result_ptrs = {}
    column_types = {}
    for col in columns:
        column_types[col] = data_frame.GetColumnType(col)
        result_ptrs[col] = data_frame.Take[column_types[col]](col)

    contents = {}
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
            contents[col] = layout

        else:  # Convert the C++ vectors to Awkward arrays
            form = ak._v2.forms.from_json(ROOT.awkward.type_to_form[col_type](0))

            if not supported(form):
                raise ak._v2._util.error(
                    NotImplementedError,
                    f"`from_rdataframe` doesn't support the {form} form yet.",
                )

            list_depth = form.purelist_depth

            data_type = cpp_type_of[form_dtype(form).name]

            # pull in the CppBuffers (after which we can import from it)
            CppBuffers = cppyy.gbl.awkward.CppBuffers[col_type]
            cpp_buffers_self = CppBuffers(result_ptrs[col])

            if isinstance(form, ak._v2.forms.NumpyForm):

                NumpyBuilder = cppyy.gbl.awkward.LayoutBuilder.Numpy[data_type]
                builder = NumpyBuilder()
                builder_type = type(builder).__cpp_name__

                cpp_buffers_self.fill_from[builder_type, col_type](
                    builder, result_ptrs[col]
                )

                names_nbytes = cpp_buffers_self.names_nbytes[builder_type](builder)
                buffers = empty_buffers(cpp_buffers_self, names_nbytes)
                cpp_buffers_self.to_char_buffers[builder_type](builder)

            elif isinstance(form, ak._v2.forms.ListOffsetForm):
                if isinstance(form.content, ak._v2.forms.NumpyForm):
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
                raise ak._v2._util.error(
                    AssertionError(f"unrecognized Form: {type(form)}")
                )

            names_nbytes = cpp_buffers_self.names_nbytes[builder_type](builder)
            buffers = empty_buffers(cpp_buffers_self, names_nbytes)
            cpp_buffers_self.to_char_buffers[builder_type](builder)

            array = ak._v2.from_buffers(
                form,
                builder.length(),
                buffers,
            )
            contents[col] = array.layout

    return ak._v2._util.wrap(
        ak._v2.contents.RecordArray(list(contents.values()), list(contents.keys())),
        highlevel=True,
    )
