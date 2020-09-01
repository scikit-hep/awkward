// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/python/startup.h"
#include "awkward/python/kernel_utils.h"
#include "awkward/python/index.h"
#include "awkward/python/identities.h"
#include "awkward/python/content.h"
#include "awkward/python/types.h"
#include "awkward/python/forms.h"
#include "awkward/python/virtual.h"
#include "awkward/python/io.h"
#include "awkward/python/partition.h"

namespace py = pybind11;
PYBIND11_MODULE(_ext, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  ////////// startup.h

  make_startup(m, "startup");

  ////////// kernel_utils.h

  make_lib_enum(m, "kernel_lib");

  ////////// index.h

  make_IndexOf<int8_t>(m,   "Index8");
  make_IndexOf<uint8_t>(m,  "IndexU8");
  make_IndexOf<int32_t>(m,  "Index32");
  make_IndexOf<uint32_t>(m, "IndexU32");
  make_IndexOf<int64_t>(m,  "Index64");

  ////////// identities.h

  make_IdentitiesOf<int32_t>(m, "Identities32");
  make_IdentitiesOf<int64_t>(m, "Identities64");

  ////////// content.h

  make_Iterator(m, "Iterator");
  make_ArrayBuilder(m, "ArrayBuilder");
  make_PersistentSharedPtr(m, "_PersistentSharedPtr");
  make_Content(m, "Content");

  make_EmptyArray(m, "EmptyArray");

  make_IndexedArrayOf<int32_t, false>(m,  "IndexedArray32");
  make_IndexedArrayOf<uint32_t, false>(m, "IndexedArrayU32");
  make_IndexedArrayOf<int64_t, false>(m,  "IndexedArray64");
  make_IndexedArrayOf<int32_t, true>(m,   "IndexedOptionArray32");
  make_IndexedArrayOf<int64_t, true>(m,   "IndexedOptionArray64");

  make_ByteMaskedArray(m,  "ByteMaskedArray");
  make_BitMaskedArray(m,   "BitMaskedArray");
  make_UnmaskedArray(m,    "UnmaskedArray");

  make_ListArrayOf<int32_t>(m,  "ListArray32");
  make_ListArrayOf<uint32_t>(m, "ListArrayU32");
  make_ListArrayOf<int64_t>(m,  "ListArray64");

  make_ListOffsetArrayOf<int32_t>(m,  "ListOffsetArray32");
  make_ListOffsetArrayOf<uint32_t>(m, "ListOffsetArrayU32");
  make_ListOffsetArrayOf<int64_t>(m,  "ListOffsetArray64");

  make_NumpyArray(m, "NumpyArray");

  make_Record(m,      "Record");
  make_RecordArray(m, "RecordArray");

  make_RegularArray(m, "RegularArray");

  make_UnionArrayOf<int8_t, int32_t>(m,  "UnionArray8_32");
  make_UnionArrayOf<int8_t, uint32_t>(m, "UnionArray8_U32");
  make_UnionArrayOf<int8_t, int64_t>(m,  "UnionArray8_64");

  make_VirtualArray(m, "VirtualArray");

  m.def("_slice_tostring", [](py::object obj) -> std::string {
    return toslice(obj).tostring();
  });

  ////////// types.h

  make_Type(m, "Type");
  make_ArrayType(m, "ArrayType");
  make_PrimitiveType(m, "PrimitiveType");
  make_RegularType(m, "RegularType");
  make_UnknownType(m, "UnknownType");
  make_ListType(m, "ListType");
  make_OptionType(m, "OptionType");
  make_UnionType(m, "UnionType");
  make_RecordType(m, "RecordType");

  ////////// forms.h

  make_Form(m, "Form");
  make_BitMaskedForm(m, "BitMaskedForm");
  make_ByteMaskedForm(m, "ByteMaskedForm");
  make_EmptyForm(m, "EmptyForm");
  make_IndexedForm(m, "IndexedForm");
  make_IndexedOptionForm(m, "IndexedOptionForm");
  make_ListForm(m, "ListForm");
  make_ListOffsetForm(m, "ListOffsetForm");
  make_NumpyForm(m, "NumpyForm");
  make_RecordForm(m, "RecordForm");
  make_RegularForm(m, "RegularForm");
  make_UnionForm(m, "UnionForm");
  make_UnmaskedForm(m, "UnmaskedForm");
  make_VirtualForm(m, "VirtualForm");

  ////////// virtual.h

  make_PyArrayGenerator(m, "ArrayGenerator");
  make_SliceGenerator(m, "SliceGenerator");
  make_PyArrayCache(m, "ArrayCache");

  ////////// io.h

  make_fromjson(m, "fromjson");
  make_fromroot_nestedvector(m, "fromroot_nestedvector");

  ////////// partition.h

  make_PartitionedArray(m, "PartitionedArray");
  make_IrregularlyPartitionedArray(m, "IrregularlyPartitionedArray");

}
