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


def from_rdataframe(data_frame, column):
    def _wrap_as_record_array(array):
        layout = array.layout if isinstance(array, ak._v2.highlevel.Array) else array
        return ak._v2._util.wrap(
            ak._v2.contents.RecordArray(
                fields=[column],
                contents=[layout],
            ),
            highlevel=True,
        )

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)
    form_str = ROOT.awkward.type_to_form[column_type](0)

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)

    if form_str.startswith("{"):
        form = ak._v2.forms.from_json(form_str)
        list_depth = form.purelist_depth
        if list_depth > 4:
            raise ak._v2._util.error(
                NotImplementedError(
                    "Retrieving arbitrary depth nested containers is not implemented yet."
                )
            )

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

        if not supported(form):
            raise ak._v2._util.error(NotImplementedError)

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

        data_type = cpp_type_of[form_dtype(form).name]

        # pull in the CppBuffers (after which we can import from it)
        CppBuffers = cppyy.gbl.awkward.CppBuffers[column_type]
        cpp_buffers_self = CppBuffers(result_ptrs)

        if isinstance(form, ak._v2.forms.NumpyForm):

            NumpyBuilder = cppyy.gbl.awkward.LayoutBuilder.Numpy[data_type]
            builder = NumpyBuilder()
            builder_type = type(builder).__cpp_name__

            cpp_buffers_self.fill_from[builder_type](builder)

        elif isinstance(form, ak._v2.forms.ListOffsetForm) and isinstance(
            form.content, ak._v2.forms.NumpyForm
        ):
            # NOTE: list_depth == 2 or 1 if its the list of strings
            ListOffsetBuilder = cppyy.gbl.awkward.LayoutBuilder.ListOffset[
                "int64_t",
                f"awkward::LayoutBuilder::Numpy<{data_type}",
            ]
            builder = ListOffsetBuilder()
            builder_type = type(builder).__cpp_name__

            cpp_buffers_self.fill_offsets_and_flatten_2[builder_type](builder)

        elif list_depth == 3:
            ListOffsetBuilder = cppyy.gbl.awkward.LayoutBuilder.ListOffset[
                "int64_t",
                f"awkward::LayoutBuilder::ListOffset<int64_t, awkward::LayoutBuilder::Numpy<{data_type}>",
            ]
            builder = ListOffsetBuilder()
            builder_type = type(builder).__cpp_name__

            cpp_buffers_self.fill_offsets_and_flatten_3[builder_type](builder)

        else:
            ListOffsetBuilder = cppyy.gbl.awkward.LayoutBuilder.ListOffset[
                "int64_t",
                f"awkward::LayoutBuilder::ListOffset<int64_t, awkward::LayoutBuilder::ListOffset<int64_t, awkward::LayoutBuilder::Numpy<{data_type}>>",
            ]
            builder = ListOffsetBuilder()
            builder_type = type(builder).__cpp_name__

            cpp_buffers_self.fill_offsets_and_flatten_4[builder_type](builder)

        names_nbytes = cpp_buffers_self.names_nbytes[builder_type](builder)
        buffers = empty_buffers(cpp_buffers_self, names_nbytes)
        cpp_buffers_self.to_char_buffers[builder_type, data_type](builder)

        array = ak._v2.from_buffers(
            form,
            builder.length(),
            buffers,
        )
        return _wrap_as_record_array(array)

    elif form_str == "awkward type":

        # ROOT::RDF::RResultPtr<T>::begin Returns an iterator to the beginning of
        # the contained object if this makes sense, throw a compilation error otherwise.
        #
        # Does not trigger event loop and execution of all actions booked in
        # the associated RLoopManager.
        lookup = result_ptrs.begin().lookup()
        generator = lookup[column].generator
        layout = generator.tolayout(lookup[column], 0, ())

        return _wrap_as_record_array(layout)
    else:
        raise ak._v2._util.error(NotImplementedError)
