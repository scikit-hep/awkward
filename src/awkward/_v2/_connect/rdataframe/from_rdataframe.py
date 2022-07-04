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

cppyy.include("rdataframe_jagged_builders.h")


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

    def _wrap_as_record_array(columns, arrays):

        for array in arrays:
            layout = (
                array.layout if isinstance(array, ak._v2.highlevel.Array) else array
            )

        return ak._v2._util.wrap(
            ak._v2.contents.RecordArray(
                fields=[column],
                contents=[layout],
            ),
            highlevel=True,
        )

    def get_layout(form_str, column_type, result_ptr):
        if form_str.startswith("{"):
            form = ak._v2.forms.from_json(form_str)
            list_depth = form.purelist_depth
            if list_depth > 3:
                raise ak._v2._util.error(NotImplementedError)

            if not supported(form):
                raise ak._v2._util.error(NotImplementedError)

            dtype = form_dtype(form)
            buffers = {}
            depths = 0
            offsets_length = 0

            # pull in the CppBuffers (after which we can import from it)
            CppBuffers = cppyy.gbl.awkward.CppBuffers[
                column_type, cpp_type_of[dtype.name]
            ]
            cpp_buffers_self = CppBuffers(result_ptr)

            if isinstance(form, ak._v2.forms.NumpyForm):

                distance = CppBuffers.result_distance(cpp_buffers_self)
                data = ak.nplike.numpy.empty(distance, dtype)
                CppBuffers.fill_data_array(
                    cpp_buffers_self, data.ctypes.data_as(ctypes.c_void_p)
                )
                layout = ak._v2.contents.numpyarray.NumpyArray(
                    data,
                    parameters=form.parameters,
                )
                return layout

            elif isinstance(form, ak._v2.forms.ListOffsetForm) and isinstance(
                form.content, ak._v2.forms.NumpyForm
            ):
                # list_depth == 2 or 1 if its the list of strings
                # copy data from RDF and make nested offsets
                depths, offsets_length = CppBuffers.offsets_and_flatten_2(
                    cpp_buffers_self
                )

            elif list_depth == 3:
                depths, offsets_length = CppBuffers.offsets_and_flatten_3(
                    cpp_buffers_self
                )
            else:
                depths, offsets_length = CppBuffers.offsets_and_flatten_4(
                    cpp_buffers_self
                )

            for depth in range(depths):
                length = CppBuffers.offsets_length(cpp_buffers_self, depth)
                offsets = ak.nplike.numpy.empty(length, np.int64)
                CppBuffers.copy_offsets(
                    cpp_buffers_self,
                    offsets.ctypes.data_as(ctypes.c_void_p),
                    length,
                    depth,
                )
                buffers[f"node{depth}-offsets"] = offsets

            data_length = CppBuffers.data_length(cpp_buffers_self)
            data = ak.nplike.numpy.empty(data_length, dtype=dtype)
            CppBuffers.copy_data(
                cpp_buffers_self, data.ctypes.data_as(ctypes.c_void_p), data_length
            )
            buffers[f"node{depths}-data"] = data

            array = ak._v2.from_buffers(
                form,
                offsets_length - 1,
                buffers,
            )
            return array.layout

        elif form_str == "awkward type":

            # ROOT::RDF::RResultPtr<T>::begin Returns an iterator to the beginning of
            # the contained object if this makes sense, throw a compilation error otherwise.
            lookup = result_ptrs[column].begin().lookup()
            generator = lookup[column].generator
            layout = generator.tolayout(lookup[column], 0, ())
            return layout

        else:
            raise ak._v2._util.error(NotImplementedError)

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    # Register Take action for each column
    result_ptrs = {}
    columns_type = {}
    form_str = {}
    layouts = []

    for column in columns:
        columns_type[column] = data_frame_rnode.GetColumnType(column)
        result_ptrs[column] = data_frame_rnode.Take[columns_type[column]](column)
        form_str[column] = ROOT.awkward.type_to_form[columns_type[column]](0)

        layout = get_layout(form_str[column], columns_type[column], result_ptrs[column])
        layouts.append(layout)

    return _wrap_as_record_array(columns, layouts)
