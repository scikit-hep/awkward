// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/python/startup.h"
#include "awkward/python/kernel_utils.h"
#include "awkward/python/index.h"
#include "awkward/python/content.h"
#include "awkward/python/forms.h"
#include "awkward/python/io.h"
#include "awkward/python/forth.h"

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

  ////////// content.h

  make_ArrayBuilder(m, "ArrayBuilder");
  make_LayoutBuilder<int32_t, int32_t>(m, "LayoutBuilder32");
  make_LayoutBuilder<int64_t, int32_t>(m, "LayoutBuilder64");

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

  ////////// io.h

  make_fromjson(m, "fromjson");
  make_fromjsonfile(m, "fromjsonfile");
  make_fromjsonobj(m, "fromjsonobj");
  make_fromjsonobj_schema(m, "fromjsonobj_schema");
  ////////// forth.h

  make_ForthMachineOf<int32_t, int32_t>(m, "ForthMachine32");
  make_ForthMachineOf<int64_t, int32_t>(m, "ForthMachine64");

}
