// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <string>

#include "awkward/Slice.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/None.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/Record.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/fillable/FillableArray.h"

#include "awkward/python/boxing.h"
#include "awkward/python/slice.h"

#include "awkward/python/util.h"

template <typename T>
py::object getitem(const T& self, const py::object& obj) {
  if (py::isinstance<py::int_>(obj)) {
    return box(self.getitem_at(obj.cast<int64_t>()));
  }
  if (py::isinstance<py::slice>(obj)) {
    py::object pystep = obj.attr("step");
    if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||  pystep.is(py::none())) {
      int64_t start = ak::Slice::none();
      int64_t stop = ak::Slice::none();
      py::object pystart = obj.attr("start");
      py::object pystop = obj.attr("stop");
      if (!pystart.is(py::none())) {
        start = pystart.cast<int64_t>();
      }
      if (!pystop.is(py::none())) {
        stop = pystop.cast<int64_t>();
      }
      return box(self.getitem_range(start, stop));
    }
    // NOTE: control flow can pass through here; don't make the last line an 'else'!
  }
  if (py::isinstance<py::str>(obj)) {
    return box(self.getitem_field(obj.cast<std::string>()));
  }
  if (!py::isinstance<py::tuple>(obj)  &&  py::isinstance<py::iterable>(obj)) {
    std::vector<std::string> strings;
    bool all_strings = true;
    for (auto x : obj) {
      if (py::isinstance<py::str>(x)) {
        strings.push_back(x.cast<std::string>());
      }
      else {
        all_strings = false;
        break;
      }
    }
    if (all_strings  &&  !strings.empty()) {
      return box(self.getitem_fields(strings));
    }
    // NOTE: control flow can pass through here; don't make the last line an 'else'!
  }
  return box(self.getitem(toslice(obj)));
}

ak::util::Parameters dict2parameters(const py::object& in) {
  ak::util::Parameters out;
  if (in.is(py::none())) {
    // None is equivalent to an empty dict
  }
  else if (py::isinstance<py::dict>(in)) {
    for (auto pair : in.cast<py::dict>()) {
      std::string key = pair.first.cast<std::string>();
      py::object value = py::module::import("json").attr("dumps")(pair.second);
      out[key] = value.cast<std::string>();
    }
  }
  else {
    throw std::invalid_argument("type parameters must be a dict (or None)");
  }
  return out;
}

py::dict parameters2dict(const ak::util::Parameters& in) {
  py::dict out;
  for (auto pair : in) {
    std::string cppkey = pair.first;
    std::string cppvalue = pair.second;
    py::str pykey(PyUnicode_DecodeUTF8(cppkey.data(), cppkey.length(), "surrogateescape"));
    py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
    out[pykey] = py::module::import("json").attr("loads")(pyvalue);
  }
  return out;
}

template py::object getitem(const ak::EmptyArray& self, const py::object& obj);
template py::object getitem(const ak::IndexedArray32& self, const py::object& obj);
template py::object getitem(const ak::IndexedArrayU32& self, const py::object& obj);
template py::object getitem(const ak::IndexedArray64& self, const py::object& obj);
template py::object getitem(const ak::IndexedOptionArray32& self, const py::object& obj);
template py::object getitem(const ak::IndexedOptionArray64& self, const py::object& obj);
template py::object getitem(const ak::ListArray32& self, const py::object& obj);
template py::object getitem(const ak::ListArrayU32& self, const py::object& obj);
template py::object getitem(const ak::ListArray64& self, const py::object& obj);
template py::object getitem(const ak::ListOffsetArray32& self, const py::object& obj);
template py::object getitem(const ak::ListOffsetArrayU32& self, const py::object& obj);
template py::object getitem(const ak::ListOffsetArray64& self, const py::object& obj);
template py::object getitem(const ak::NumpyArray& self, const py::object& obj);
template py::object getitem(const ak::Record& self, const py::object& obj);
template py::object getitem(const ak::RecordArray& self, const py::object& obj);
template py::object getitem(const ak::RegularArray& self, const py::object& obj);
template py::object getitem(const ak::UnionArray8_32& self, const py::object& obj);
template py::object getitem(const ak::UnionArray8_U32& self, const py::object& obj);
template py::object getitem(const ak::UnionArray8_64& self, const py::object& obj);

template py::object getitem(const ak::FillableArray& self, const py::object& obj);
