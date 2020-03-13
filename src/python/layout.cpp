// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>

#include "awkward/python/index.h"
#include "awkward/python/identities.h"
#include "awkward/python/content.h"

namespace py = pybind11;
PYBIND11_MODULE(layout, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  /////////////////////////////////////////////////////////////// index.h

  make_IndexOf<int8_t>(m,   "Index8");
  make_IndexOf<uint8_t>(m,  "IndexU8");
  make_IndexOf<int32_t>(m,  "Index32");
  make_IndexOf<uint32_t>(m, "IndexU32");
  make_IndexOf<int64_t>(m,  "Index64");

  /////////////////////////////////////////////////////////////// identities.h

  make_IdentitiesOf<int32_t>(m, "Identities32");
  make_IdentitiesOf<int64_t>(m, "Identities64");

  /////////////////////////////////////////////////////////////// content.h

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

  m.def("_slice_tostring", [](py::object obj) -> std::string {
    return toslice(obj).tostring();
  });
}
