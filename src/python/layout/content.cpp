// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>

#include "awkward/python/identities.h"
#include "awkward/python/util.h"

#include "awkward/python/content.h"

/////////////////////////////////////////////////////////////// boxing

py::object box(const std::shared_ptr<ak::Content>& content) {
  // scalars
  if (ak::None* raw = dynamic_cast<ak::None*>(content.get())) {
    return py::none();
  }
  else if (ak::Record* raw = dynamic_cast<ak::Record*>(content.get())) {
    return py::cast(*raw);
  }
  // scalar or array
  else if (ak::NumpyArray* raw = dynamic_cast<ak::NumpyArray*>(content.get())) {
    if (raw->isscalar()) {
      return py::array(py::buffer_info(
        raw->byteptr(),
        raw->itemsize(),
        raw->format(),
        raw->ndim(),
        raw->shape(),
        raw->strides()
      )).attr("item")();
    }
    else {
      return py::cast(*raw);
    }
  }
  // arrays
  else if (ak::EmptyArray* raw = dynamic_cast<ak::EmptyArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArray32* raw = dynamic_cast<ak::IndexedArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArrayU32* raw = dynamic_cast<ak::IndexedArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedArray64* raw = dynamic_cast<ak::IndexedArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedOptionArray32* raw = dynamic_cast<ak::IndexedOptionArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::IndexedOptionArray64* raw = dynamic_cast<ak::IndexedOptionArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ByteMaskedArray* raw = dynamic_cast<ak::ByteMaskedArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::BitMaskedArray* raw = dynamic_cast<ak::BitMaskedArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnmaskedArray* raw = dynamic_cast<ak::UnmaskedArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArray32* raw = dynamic_cast<ak::ListArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArrayU32* raw = dynamic_cast<ak::ListArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListArray64* raw = dynamic_cast<ak::ListArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray32* raw = dynamic_cast<ak::ListOffsetArray32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArrayU32* raw = dynamic_cast<ak::ListOffsetArrayU32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListOffsetArray64* raw = dynamic_cast<ak::ListOffsetArray64*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::RecordArray* raw = dynamic_cast<ak::RecordArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::RegularArray* raw = dynamic_cast<ak::RegularArray*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_32* raw = dynamic_cast<ak::UnionArray8_32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_U32* raw = dynamic_cast<ak::UnionArray8_U32*>(content.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionArray8_64* raw = dynamic_cast<ak::UnionArray8_64*>(content.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Content subtype");
  }
}

py::object box(const std::shared_ptr<ak::Identities>& identities) {
  if (identities.get() == nullptr) {
    return py::none();
  }
  else if (ak::Identities32* raw = dynamic_cast<ak::Identities32*>(identities.get())) {
    return py::cast(*raw);
  }
  else if (ak::Identities64* raw = dynamic_cast<ak::Identities64*>(identities.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Identities subtype");
  }
}

std::shared_ptr<ak::Content> unbox_content(const py::handle& obj) {
  try {
    obj.cast<ak::Record*>();
    throw std::invalid_argument("content argument must be a Content subtype (excluding Record)");
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::NumpyArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::EmptyArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedArrayU32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedOptionArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::IndexedOptionArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ByteMaskedArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::BitMaskedArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnmaskedArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListArrayU32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListOffsetArray32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListOffsetArrayU32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListOffsetArray64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RecordArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RegularArray*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionArray8_32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionArray8_U32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionArray8_64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("content argument must be a Content subtype");
}

std::shared_ptr<ak::Identities> unbox_identities(const py::handle& obj) {
  try {
    return obj.cast<ak::Identities32*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::Identities64*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("id argument must be an Identities subtype");
}

std::shared_ptr<ak::Identities> unbox_identities_none(const py::handle& obj) {
  if (obj.is(py::none())) {
    return ak::Identities::none();
  }
  else {
    return unbox_identities(obj);
  }
}

/////////////////////////////////////////////////////////////// slicing

bool handle_as_numpy(const std::shared_ptr<ak::Content>& content) {
  // if (content.get()->parameter_equals("__array__", "\"string\"")  ||  content.get()->parameter_equals("__array__", "\"bytestring\"")) {
  //   return true;
  // }
  if (ak::NumpyArray* raw = dynamic_cast<ak::NumpyArray*>(content.get())) {
    return true;
  }
  else if (ak::EmptyArray* raw = dynamic_cast<ak::EmptyArray*>(content.get())) {
    return true;
  }
  else if (ak::RegularArray* raw = dynamic_cast<ak::RegularArray*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArray32* raw = dynamic_cast<ak::IndexedArray32*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArrayU32* raw = dynamic_cast<ak::IndexedArrayU32*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::IndexedArray64* raw = dynamic_cast<ak::IndexedArray64*>(content.get())) {
    return handle_as_numpy(raw->content());
  }
  else if (ak::UnionArray8_32* raw = dynamic_cast<ak::UnionArray8_32*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else if (ak::UnionArray8_U32* raw = dynamic_cast<ak::UnionArray8_U32*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else if (ak::UnionArray8_64* raw = dynamic_cast<ak::UnionArray8_64*>(content.get())) {
    std::shared_ptr<ak::Content> first = raw->content(0);
    for (int64_t i = 1;  i < raw->numcontents();  i++) {
      if (!first.get()->mergeable(raw->content(i), false)) {
        return false;
      }
    }
    return handle_as_numpy(first);
  }
  else {
    return false;
  }
}

void toslice_part(ak::Slice& slice, py::object obj) {
  int64_t length_before = slice.length();

  if (py::isinstance<py::int_>(obj)) {
    // FIXME: what happens if you give this a Numpy integer? a Numpy 0-dimensional array?
    slice.append(std::make_shared<ak::SliceAt>(obj.cast<int64_t>()));
  }

  else if (py::isinstance<py::slice>(obj)) {
    py::object pystart = obj.attr("start");
    py::object pystop = obj.attr("stop");
    py::object pystep = obj.attr("step");
    int64_t start = ak::Slice::none();
    int64_t stop = ak::Slice::none();
    int64_t step = 1;
    if (!pystart.is(py::none())) {
      start = pystart.cast<int64_t>();
    }
    if (!pystop.is(py::none())) {
      stop = pystop.cast<int64_t>();
    }
    if (!pystep.is(py::none())) {
      step = pystep.cast<int64_t>();
    }
    if (step == 0) {
      throw std::invalid_argument("slice step must not be 0");
    }
    slice.append(std::make_shared<ak::SliceRange>(start, stop, step));
  }

#if PY_MAJOR_VERSION >= 3
  else if (py::isinstance<py::ellipsis>(obj)) {
    slice.append(std::make_shared<ak::SliceEllipsis>());
  }
#endif

  else if (obj.is(py::module::import("numpy").attr("newaxis"))) {
    slice.append(std::make_shared<ak::SliceNewAxis>());
  }

  else if (py::isinstance<py::str>(obj)) {
    slice.append(std::make_shared<ak::SliceField>(obj.cast<std::string>()));
  }

  else if (py::isinstance<py::iterable>(obj)) {
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
      slice.append(std::make_shared<ak::SliceFields>(strings));
    }
    else {
      std::shared_ptr<ak::Content> content(nullptr);

      if (py::isinstance(obj, py::module::import("numpy").attr("ma").attr("MaskedArray"))) {
        content = unbox_content(py::module::import("awkward1").attr("fromnumpy")(obj, false, false));
      }
      else if (py::isinstance(obj, py::module::import("numpy").attr("ndarray"))) {
        // content = nullptr!
      }
      else if (py::isinstance<ak::Content>(obj)) {
        content = unbox_content(obj);
      }
      else if (py::isinstance<ak::ArrayBuilder>(obj)) {
        content = unbox_content(obj.attr("snapshot")());
      }
      else if (py::isinstance(obj, py::module::import("awkward1").attr("Array"))) {
        content = unbox_content(obj.attr("layout"));
      }
      else if (py::isinstance(obj, py::module::import("awkward1").attr("ArrayBuilder"))) {
        content = unbox_content(obj.attr("snapshot")().attr("layout"));
      }
      else {
        bool bad = false;
        try {
          obj = py::module::import("numpy").attr("asarray")(obj);
        }
        catch (py::error_already_set& exc) {
          exc.restore();
          PyErr_Clear();
          bad = true;
        }
        if (!bad) {
          py::array array = obj.cast<py::array>();
          py::buffer_info info = array.request();
          if (info.format.compare("O") == 0) {
            bad = true;
          }
        }
        if (bad) {
          content = unbox_content(py::module::import("awkward1").attr("fromiter")(obj, false));
        }
      }

      if (content.get() != nullptr  &&  !handle_as_numpy(content)) {
        if (content.get()->parameter_equals("__array__", "\"string\"")  ||  content.get()->parameter_equals("__array__", "\"bytestring\"")) {
          obj = box(content);
          obj = py::module::import("awkward1").attr("tolist")(obj);
          std::vector<std::string> strings;
          for (auto x : obj) {
            strings.push_back(x.cast<std::string>());
          }
          slice.append(std::make_shared<ak::SliceFields>(strings));
        }
        else {
          slice.append(content.get()->asslice());
        }
      }
      else {
        py::array array = obj.cast<py::array>();
        if (array.ndim() == 0) {
          throw std::invalid_argument("arrays used as an index must have at least one dimension");
        }

        py::buffer_info info = array.request();
        if (info.format.compare("?") == 0) {
          py::object nonzero_tuple = py::module::import("numpy").attr("nonzero")(array);
          for (auto x : nonzero_tuple.cast<py::tuple>()) {
            py::object intarray_object = py::module::import("numpy").attr("asarray")(x.cast<py::object>(), py::module::import("numpy").attr("int64"));
            py::array intarray = intarray_object.cast<py::array>();
            py::buffer_info intinfo = intarray.request();
            std::vector<int64_t> shape;
            std::vector<int64_t> strides;
            for (ssize_t i = 0;  i < intinfo.ndim;  i++) {
              shape.push_back((int64_t)intinfo.shape[i]);
              strides.push_back((int64_t)intinfo.strides[i] / sizeof(int64_t));
            }
            ak::Index64 index(std::shared_ptr<int64_t>(reinterpret_cast<int64_t*>(intinfo.ptr), pyobject_deleter<int64_t>(intarray.ptr())), 0, shape[0]);
            slice.append(std::make_shared<ak::SliceArray64>(index, shape, strides, true));
          }
        }

        else {
          ssize_t flatlen = 1;
          for (auto x : info.shape) {
            flatlen *= x;
          }
          std::string format(info.format);
          format.erase(0, format.find_first_not_of("@=<>!"));
          if (py::isinstance<py::array>(obj) &&
              format.compare("c") != 0       &&
              format.compare("b") != 0       &&
              format.compare("B") != 0       &&
              format.compare("h") != 0       &&
              format.compare("H") != 0       &&
              format.compare("i") != 0       &&
              format.compare("I") != 0       &&
              format.compare("l") != 0       &&
              format.compare("L") != 0       &&
              format.compare("q") != 0       &&
              format.compare("Q") != 0       &&
              flatlen != 0) {
            throw std::invalid_argument("arrays used as an index must be integer or boolean");
          }

          py::object intarray_object = py::module::import("numpy").attr("asarray")(array, py::module::import("numpy").attr("int64"));
          py::array intarray = intarray_object.cast<py::array>();
          py::buffer_info intinfo = intarray.request();
          std::vector<int64_t> shape;
          std::vector<int64_t> strides;
          for (ssize_t i = 0;  i < intinfo.ndim;  i++) {
            shape.push_back((int64_t)intinfo.shape[i]);
            strides.push_back((int64_t)intinfo.strides[i] / (int64_t)sizeof(int64_t));
          }
          ak::Index64 index(std::shared_ptr<int64_t>(reinterpret_cast<int64_t*>(intinfo.ptr), pyobject_deleter<int64_t>(intarray.ptr())), 0, shape[0]);
          slice.append(std::make_shared<ak::SliceArray64>(index, shape, strides, false));
        }
      }
    }
  }

  else {
    throw std::invalid_argument("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), and integer or boolean arrays (possibly jagged) are valid indices");
  }
}

ak::Slice toslice(py::object obj) {
  ak::Slice out;
  if (py::isinstance<py::tuple>(obj)) {
    for (auto x : obj.cast<py::tuple>()) {
      toslice_part(out, x.cast<py::object>());
    }
  }
  else {
    toslice_part(out, obj);
  }
  out.become_sealed();
  return out;
}

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

/////////////////////////////////////////////////////////////// ArrayBuilder

void builder_fromiter(ak::ArrayBuilder& self, const py::handle& obj) {
  if (obj.is(py::none())) {
    self.null();
  }
  else if (py::isinstance<py::bool_>(obj)) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance<py::int_>(obj)) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance<py::float_>(obj)) {
    self.real(obj.cast<double>());
  }
  else if (py::isinstance<py::bytes>(obj)) {
    self.bytestring(obj.cast<std::string>());
  }
  else if (py::isinstance<py::str>(obj)) {
    self.string(obj.cast<std::string>());
  }
  else if (py::isinstance<py::tuple>(obj)) {
    py::tuple tup = obj.cast<py::tuple>();
    self.begintuple(tup.size());
    for (size_t i = 0;  i < tup.size();  i++) {
      self.index((int64_t)i);
      builder_fromiter(self, tup[i]);
    }
    self.endtuple();
  }
  else if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    self.beginrecord();
    for (auto pair : dict) {
      if (!py::isinstance<py::str>(pair.first)) {
        throw std::invalid_argument("keys of dicts in 'fromiter' must all be strings");
      }
      std::string key = pair.first.cast<std::string>();
      self.field_check(key.c_str());
      builder_fromiter(self, pair.second);
    }
    self.endrecord();
  }
  else if (py::isinstance<py::iterable>(obj)) {
    py::iterable seq = obj.cast<py::iterable>();
    self.beginlist();
    for (auto x : seq) {
      builder_fromiter(self, x);
    }
    self.endlist();
  }
  else if (py::isinstance<py::array>(obj)) {
    py::iterable seq = obj.attr("tolist")().cast<py::iterable>();
    self.beginlist();
    for (auto x : seq) {
      builder_fromiter(self, x);
    }
    self.endlist();
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("bool_"))) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("integer"))) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("floating"))) {
    self.real(obj.cast<double>());
  }
  else {
    throw std::invalid_argument(std::string("cannot convert ") + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ") + obj.attr("__class__").attr("__name__").cast<std::string>() + std::string(") to an array element"));
  }
}

py::class_<ak::ArrayBuilder> make_ArrayBuilder(const py::handle& m, const std::string& name) {
  return (py::class_<ak::ArrayBuilder>(m, name.c_str())
      .def(py::init([](int64_t initial, double resize) -> ak::ArrayBuilder {
        return ak::ArrayBuilder(ak::ArrayBuilderOptions(initial, resize));
      }), py::arg("initial") = 1024, py::arg("resize") = 2.0)
      .def_property_readonly("_ptr", [](const ak::ArrayBuilder* self) -> size_t { return reinterpret_cast<size_t>(self); })
      .def("__repr__", &ak::ArrayBuilder::tostring)
      .def("__len__", &ak::ArrayBuilder::length)
      .def("clear", &ak::ArrayBuilder::clear)
      .def("type", &ak::ArrayBuilder::type)
      .def("snapshot", [](const ak::ArrayBuilder& self) -> py::object {
        return box(self.snapshot());
      })
      .def("__getitem__", &getitem<ak::ArrayBuilder>)
      .def("__iter__", [](const ak::ArrayBuilder& self) -> ak::Iterator {
        return ak::Iterator(self.snapshot());
      })
      .def("null", &ak::ArrayBuilder::null)
      .def("boolean", &ak::ArrayBuilder::boolean)
      .def("integer", &ak::ArrayBuilder::integer)
      .def("real", &ak::ArrayBuilder::real)
      .def("bytestring", [](ak::ArrayBuilder& self, const py::bytes& x) -> void {
        self.bytestring(x.cast<std::string>());
      })
      .def("string", [](ak::ArrayBuilder& self, const py::str& x) -> void {
        self.string(x.cast<std::string>());
      })
      .def("beginlist", &ak::ArrayBuilder::beginlist)
      .def("endlist", &ak::ArrayBuilder::endlist)
      .def("begintuple", &ak::ArrayBuilder::begintuple)
      .def("index", &ak::ArrayBuilder::index)
      .def("endtuple", &ak::ArrayBuilder::endtuple)
      .def("beginrecord", [](ak::ArrayBuilder& self, const py::object& name) -> void {
        if (name.is(py::none())) {
          self.beginrecord();
        }
        else {
          std::string cppname = name.cast<std::string>();
          self.beginrecord_check(cppname.c_str());
        }
      }, py::arg("name") = py::none())
      .def("field", [](ak::ArrayBuilder& self, const std::string& x) -> void {
        self.field_check(x);
      })
      .def("endrecord", &ak::ArrayBuilder::endrecord)
      .def("append", [](ak::ArrayBuilder& self, const std::shared_ptr<ak::Content>& array, int64_t at) {
        self.append(array, at);
      })
      .def("extend", [](ak::ArrayBuilder& self, const std::shared_ptr<ak::Content>& array) {
        self.extend(array);
      })
      .def("fromiter", &builder_fromiter)
  );
}

/////////////////////////////////////////////////////////////// Iterator

py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>> make_Iterator(const py::handle& m, const std::string& name) {
  auto next = [](ak::Iterator& iterator) -> py::object {
    if (iterator.isdone()) {
      throw py::stop_iteration();
    }
    return box(iterator.next());
  };

  return (py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>>(m, name.c_str())
      .def(py::init([](const py::object& content) -> ak::Iterator {
        return ak::Iterator(unbox_content(content));
      }))
      .def("__repr__", &ak::Iterator::tostring)
      .def("__next__", next)
      .def("next", next)
      .def("__iter__", [](const py::object& self) -> py::object { return self; })
  );
}

/////////////////////////////////////////////////////////////// Content

PersistentSharedPtr::PersistentSharedPtr(const std::shared_ptr<ak::Content>& ptr)
    : ptr_(ptr) { }

py::object PersistentSharedPtr::layout() const {
  return box(ptr_);
}

size_t PersistentSharedPtr::ptr() const {
  return reinterpret_cast<size_t>(&ptr_);
}

py::class_<PersistentSharedPtr> make_PersistentSharedPtr(const py::handle& m, const std::string& name) {
  return py::class_<PersistentSharedPtr>(m, name.c_str())
             .def("layout", &PersistentSharedPtr::layout)
             .def("ptr", &PersistentSharedPtr::ptr);
}

py::class_<ak::Content, std::shared_ptr<ak::Content>> make_Content(const py::handle& m, const std::string& name) {
  return py::class_<ak::Content, std::shared_ptr<ak::Content>>(m, name.c_str());
}

template <typename T>
py::tuple identity(const T& self) {
  if (self.identities().get() == nullptr) {
    throw std::invalid_argument(self.classname() + std::string(" instance has no associated identities (use 'setidentities' to assign one to the array it is in)"));
  }
  ak::Identities::FieldLoc fieldloc = self.identities().get()->fieldloc();
  if (self.isscalar()) {
    py::tuple out((size_t)(self.identities().get()->width()) + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.identities().get()->width();  i++) {
      out[j] = py::cast(self.identities().get()->value(0, i));
      j++;
      for (auto pair : fieldloc) {
        if (pair.first == i) {
          out[j] = py::cast(pair.second);
          j++;
        }
      }
    }
    return out;
  }
  else {
    py::tuple out((size_t)(self.identities().get()->width() - 1) + fieldloc.size());
    size_t j = 0;
    for (int64_t i = 0;  i < self.identities().get()->width();  i++) {
      if (i < self.identities().get()->width() - 1) {
        out[j] = py::cast(self.identities().get()->value(0, i));
        j++;
      }
      for (auto pair : fieldloc) {
        if (pair.first == i) {
          out[j] = py::cast(pair.second);
          j++;
        }
      }
    }
    return out;
  }
}

template <typename T>
std::string repr(const T& self) {
  return self.tostring();
}

template <typename T>
int64_t len(const T& self) {
  return self.length();
}

template <typename T>
ak::Iterator iter(const T& self) {
  return ak::Iterator(self.shallow_copy());
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

template <typename T>
py::dict getparameters(const T& self) {
  return parameters2dict(self.parameters());
}

template <typename T>
py::object parameter(const T& self, const std::string& key) {
  std::string cppvalue = self.parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
py::object purelist_parameter(const T& self, const std::string& key) {
  std::string cppvalue = self.purelist_parameter(key);
  py::str pyvalue(PyUnicode_DecodeUTF8(cppvalue.data(), cppvalue.length(), "surrogateescape"));
  return py::module::import("json").attr("loads")(pyvalue);
}

template <typename T>
void setparameters(T& self, const py::object& parameters) {
  self.setparameters(dict2parameters(parameters));
}

template <typename T>
void setparameter(T& self, const std::string& key, const py::object& value) {
  py::object valuestr = py::module::import("json").attr("dumps")(value);
  self.setparameter(key, valuestr.cast<std::string>());
}

int64_t check_maxdecimals(const py::object& maxdecimals) {
  if (maxdecimals.is(py::none())) {
    return -1;
  }
  try {
    return maxdecimals.cast<int64_t>();
  }
  catch (py::cast_error err) {
    throw std::invalid_argument("maxdecimals must be None or an integer");
  }
}

template <typename T>
std::string tojson_string(const T& self, bool pretty, const py::object& maxdecimals) {
  return self.tojson(pretty, check_maxdecimals(maxdecimals));
}

template <typename T>
void tojson_file(const T& self, const std::string& destination, bool pretty, py::object maxdecimals, int64_t buffersize) {
#ifdef _MSC_VER
  FILE* file;
  if (fopen_s(&file, destination.c_str(), "wb") != 0) {
#else
  FILE* file = fopen(destination.c_str(), "wb");
  if (file == nullptr) {
#endif
    throw std::invalid_argument(std::string("file \"") + destination + std::string("\" could not be opened for writing"));
  }
  try {
    self.tojson(file, pretty, check_maxdecimals(maxdecimals), buffersize);
  }
  catch (...) {
    fclose(file);
    throw;
  }
  fclose(file);
}

template <typename T>
py::class_<T, std::shared_ptr<T>, ak::Content> content_methods(py::class_<T, std::shared_ptr<T>, ak::Content>& x) {
  return x.def("__repr__", &repr<T>)
          .def_property("identities", [](const T& self) -> py::object { return box(self.identities()); }, [](T& self, const py::object& identities) -> void { self.setidentities(unbox_identities_none(identities)); })
          .def("setidentities", [](T& self, const py::object& identities) -> void {
           self.setidentities(unbox_identities_none(identities));
          })
          .def("setidentities", [](T& self) -> void {
            self.setidentities();
          })
          .def_property("parameters", &getparameters<T>, &setparameters<T>)
          .def("setparameter", &setparameter<T>)
          .def("parameter", &parameter<T>)
          .def("purelist_parameter", &purelist_parameter<T>)
          .def("type", [](const T& self, const std::map<std::string, std::string>& typestrs) -> std::shared_ptr<ak::Type> {
            return self.type(typestrs);
          })
          .def("__len__", &len<T>)
          .def("__getitem__", &getitem<T>)
          .def("__iter__", &iter<T>)
          .def("tojson", &tojson_string<T>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
          .def("tojson", &tojson_file<T>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)
          .def_property_readonly("nbytes", &T::nbytes)
          .def("deep_copy", &T::deep_copy, py::arg("copyarrays") = true, py::arg("copyindexes") = true, py::arg("copyidentities") = true)
          .def_property_readonly("identity", &identity<T>)
          .def_property_readonly("numfields", &T::numfields)
          .def("fieldindex", &T::fieldindex)
          .def("key", &T::key)
          .def("haskey", &T::haskey)
          .def("keys", &T::keys)
          .def_property_readonly("purelist_isregular", &T::purelist_isregular)
          .def_property_readonly("purelist_depth", &T::purelist_depth)
          .def("getitem_nothing", &T::getitem_nothing)
          .def_property_readonly("_persistent_shared_ptr", [](std::shared_ptr<ak::Content>& self) -> PersistentSharedPtr {
            return PersistentSharedPtr(self);
          })

          // operations
          .def("validityerror", [](const T& self) -> py::object {
            std::string out = self.validityerror(std::string("layout"));
            if (out.empty()) {
              return py::none();
            }
            else {
              py::str pyvalue(PyUnicode_DecodeUTF8(out.data(), out.length(), "surrogateescape"));
              return pyvalue;
            }
          })
          .def("num", [](const T& self, int64_t axis) -> py::object {
            return box(self.num(axis, 0));
          }, py::arg("axis") = 1)
          .def("flatten", [](const T& self, int64_t axis) -> py::object {
            std::pair<ak::Index64, std::shared_ptr<ak::Content>> pair = self.offsets_and_flattened(axis, 0);
            return box(pair.second);
          }, py::arg("axis") = 1)
          .def("offsets_and_flatten", [](const T& self, int64_t axis) -> py::object {
            std::pair<ak::Index64, std::shared_ptr<ak::Content>> pair = self.offsets_and_flattened(axis, 0);
            return py::make_tuple(py::cast(pair.first), box(pair.second));
          }, py::arg("axis") = 1)
          .def("rpad", [](const T&self, int64_t length, int64_t axis) -> py::object {
            return box(self.rpad(length, axis, 0));
          })
          .def("rpad_and_clip", [](const T&self, int64_t length, int64_t axis) -> py::object {
            return box(self.rpad_and_clip(length, axis, 0));
          })
          .def("mergeable", [](const T& self, const py::object& other, bool mergebool) -> bool {
            return self.mergeable(unbox_content(other), mergebool);
          }, py::arg("other"), py::arg("mergebool") = false)
          .def("merge", [](const T& self, const py::object& other) -> py::object {
            return box(self.merge(unbox_content(other)));
          })
          .def("merge_as_union", [](const T& self, const py::object& other) -> py::object {
            return box(self.merge_as_union(unbox_content(other)));
          })
          .def("count", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerCount reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = false, py::arg("keepdims") = false)
          .def("count_nonzero", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerCountNonzero reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = false, py::arg("keepdims") = false)
          .def("sum", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerSum reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = false, py::arg("keepdims") = false)
          .def("prod", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerProd reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = false, py::arg("keepdims") = false)
          .def("any", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerAny reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = false, py::arg("keepdims") = false)
          .def("all", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerAll reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = false, py::arg("keepdims") = false)
          .def("min", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerMin reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = true, py::arg("keepdims") = false)
          .def("max", [](const T& self, int64_t axis, bool mask, bool keepdims) -> py::object {
            ak::ReducerMax reducer;
            return box(self.reduce(reducer, axis, mask, keepdims));
          }, py::arg("axis") = -1, py::arg("mask") = true, py::arg("keepdims") = false)
          .def("localindex", [](const T& self, int64_t axis) -> py::object {
            return box(self.localindex(axis, 0));
          }, py::arg("axis") = 1)
          .def("choose", [](const T& self, int64_t n, bool diagonal, py::object keys, py::object parameters, int64_t axis) -> py::object {
            std::shared_ptr<ak::util::RecordLookup> recordlookup(nullptr);
            if (!keys.is(py::none())) {
              recordlookup = std::make_shared<ak::util::RecordLookup>();
              for (auto x : keys.cast<py::iterable>()) {
                recordlookup.get()->push_back(x.cast<std::string>());
              }
              if (n != recordlookup.get()->size()) {
                throw std::invalid_argument("if provided, the length of 'keys' must be 'n'");
              }
            }
            return box(self.choose(n, diagonal, recordlookup, dict2parameters(parameters), axis, 0));
          }, py::arg("n"), py::arg("diagonal") = false, py::arg("keys") = py::none(), py::arg("parameters") = py::none(), py::arg("axis") = 1)

  ;
}

/////////////////////////////////////////////////////////////// EmptyArray

py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content> make_EmptyArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content>(m, name.c_str())
      .def(py::init([](const py::object& identities, const py::object& parameters) -> ak::EmptyArray {
        return ak::EmptyArray(unbox_identities_none(identities), dict2parameters(parameters));
      }), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
      .def("toNumpyArray", [](const ak::EmptyArray& self) -> py::object {
        return box(self.toNumpyArray("d", sizeof(double)));
      })
      .def("simplify", [](const ak::EmptyArray& self) {
        return box(self.shallow_simplify());
      })
  );
}

/////////////////////////////////////////////////////////////// IndexedArray

template <typename T, bool ISOPTION>
py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content>(m, name.c_str())
      .def(py::init([](const ak::IndexOf<T>& index, const py::object& content, const py::object& identities, const py::object& parameters) -> ak::IndexedArrayOf<T, ISOPTION> {
        return ak::IndexedArrayOf<T, ISOPTION>(unbox_identities_none(identities), dict2parameters(parameters), index, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("index"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("index", &ak::IndexedArrayOf<T, ISOPTION>::index)
      .def_property_readonly("content", &ak::IndexedArrayOf<T, ISOPTION>::content)
      .def_property_readonly("isoption", &ak::IndexedArrayOf<T, ISOPTION>::isoption)
      .def("project", [](const ak::IndexedArrayOf<T, ISOPTION>& self, const py::object& mask) {
        if (mask.is(py::none())) {
          return box(self.project());
        }
        else {
          return box(self.project(mask.cast<ak::Index8>()));
        }
      }, py::arg("mask") = py::none())
      .def("bytemask", &ak::IndexedArrayOf<T, ISOPTION>::bytemask)
      .def("simplify", [](const ak::IndexedArrayOf<T, ISOPTION>& self) {
        return box(self.simplify_optiontype());
      })
  );
}

template py::class_<ak::IndexedArray32, std::shared_ptr<ak::IndexedArray32>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::IndexedArrayU32, std::shared_ptr<ak::IndexedArrayU32>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::IndexedArray64, std::shared_ptr<ak::IndexedArray64>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::IndexedOptionArray32, std::shared_ptr<ak::IndexedOptionArray32>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::IndexedOptionArray64, std::shared_ptr<ak::IndexedOptionArray64>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name);

/////////////////////////////////////////////////////////////// ByteMaskedArray

py::class_<ak::ByteMaskedArray, std::shared_ptr<ak::ByteMaskedArray>, ak::Content> make_ByteMaskedArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::ByteMaskedArray, std::shared_ptr<ak::ByteMaskedArray>, ak::Content>(m, name.c_str())
      .def(py::init([](const ak::Index8& mask, const py::object& content, bool validwhen, const py::object& identities, const py::object& parameters) -> ak::ByteMaskedArray {
        return ak::ByteMaskedArray(unbox_identities_none(identities), dict2parameters(parameters), mask, std::shared_ptr<ak::Content>(unbox_content(content)), validwhen);
      }), py::arg("mask"), py::arg("content"), py::arg("validwhen"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("mask", &ak::ByteMaskedArray::mask)
      .def_property_readonly("content", &ak::ByteMaskedArray::content)
      .def_property_readonly("validwhen", &ak::ByteMaskedArray::validwhen)
      .def("project", [](const ak::ByteMaskedArray& self, const py::object& mask) {
        if (mask.is(py::none())) {
          return box(self.project());
        }
        else {
          return box(self.project(mask.cast<ak::Index8>()));
        }
      }, py::arg("mask") = py::none())
      .def("bytemask", &ak::ByteMaskedArray::bytemask)
      .def("simplify", [](const ak::ByteMaskedArray& self) {
        return box(self.simplify_optiontype());
      })
  );
}

/////////////////////////////////////////////////////////////// BitMaskedArray

py::class_<ak::BitMaskedArray, std::shared_ptr<ak::BitMaskedArray>, ak::Content> make_BitMaskedArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::BitMaskedArray, std::shared_ptr<ak::BitMaskedArray>, ak::Content>(m, name.c_str())
      .def(py::init([](const ak::IndexU8& mask, const py::object& content, bool validwhen, int64_t length, bool lsb_order, const py::object& identities, const py::object& parameters) -> ak::BitMaskedArray {
        return ak::BitMaskedArray(unbox_identities_none(identities), dict2parameters(parameters), mask, std::shared_ptr<ak::Content>(unbox_content(content)), validwhen, length, lsb_order);
      }), py::arg("mask"), py::arg("content"), py::arg("validwhen"), py::arg("length"), py::arg("lsb_order"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("mask", &ak::BitMaskedArray::mask)
      .def_property_readonly("content", &ak::BitMaskedArray::content)
      .def_property_readonly("validwhen", &ak::BitMaskedArray::validwhen)
      .def("project", [](const ak::BitMaskedArray& self, const py::object& mask) {
        if (mask.is(py::none())) {
          return box(self.project());
        }
        else {
          return box(self.project(mask.cast<ak::Index8>()));
        }
      }, py::arg("mask") = py::none())
      .def("bytemask", &ak::BitMaskedArray::bytemask)
      .def("simplify", [](const ak::BitMaskedArray& self) {
        return box(self.simplify_optiontype());
      })
      .def("toByteMaskedArray", &ak::BitMaskedArray::toByteMaskedArray)
      .def("toIndexedOptionArray64", &ak::BitMaskedArray::toIndexedOptionArray64)
  );
}

/////////////////////////////////////////////////////////////// UnmaskedArray

py::class_<ak::UnmaskedArray, std::shared_ptr<ak::UnmaskedArray>, ak::Content> make_UnmaskedArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::UnmaskedArray, std::shared_ptr<ak::UnmaskedArray>, ak::Content>(m, name.c_str())
      .def(py::init([](const py::object& content, const py::object& identities, const py::object& parameters) -> ak::UnmaskedArray {
        return ak::UnmaskedArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("content", &ak::UnmaskedArray::content)
      .def("project", [](const ak::UnmaskedArray& self, const py::object& mask) {
        if (mask.is(py::none())) {
          return box(self.project());
        }
        else {
          return box(self.project(mask.cast<ak::Index8>()));
        }
      }, py::arg("mask") = py::none())
      .def("bytemask", &ak::UnmaskedArray::bytemask)
      .def("simplify", [](const ak::UnmaskedArray& self) {
        return box(self.simplify_optiontype());
      })
  );
}

/////////////////////////////////////////////////////////////// ListArray

template <typename T>
py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content> make_ListArrayOf(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](const ak::IndexOf<T>& starts, const ak::IndexOf<T>& stops, const py::object& content, const py::object& identities, const py::object& parameters) -> ak::ListArrayOf<T> {
        return ak::ListArrayOf<T>(unbox_identities_none(identities), dict2parameters(parameters), starts, stops, unbox_content(content));
      }), py::arg("starts"), py::arg("stops"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("starts", &ak::ListArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListArrayOf<T>::stops)
      .def_property_readonly("content", &ak::ListArrayOf<T>::content)
      .def("compact_offsets64", &ak::ListArrayOf<T>::compact_offsets64, py::arg("start_at_zero") = true)
      .def("broadcast_tooffsets64", &ak::ListArrayOf<T>::broadcast_tooffsets64)
      .def("toRegularArray", &ak::ListArrayOf<T>::toRegularArray)
      .def("simplify", [](const ak::ListArrayOf<T>& self) {
        return box(self.shallow_simplify());
      })
  );
}

template py::class_<ak::ListArray32, std::shared_ptr<ak::ListArray32>, ak::Content> make_ListArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::ListArrayU32, std::shared_ptr<ak::ListArrayU32>, ak::Content> make_ListArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::ListArray64, std::shared_ptr<ak::ListArray64>, ak::Content> make_ListArrayOf(const py::handle& m, const std::string& name);

/////////////////////////////////////////////////////////////// ListOffsetArray

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content> make_ListOffsetArrayOf(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content>(m, name.c_str())
      .def(py::init([](const ak::IndexOf<T>& offsets, const py::object& content, const py::object& identities, const py::object& parameters) -> ak::ListOffsetArrayOf<T> {
        return ak::ListOffsetArrayOf<T>(unbox_identities_none(identities), dict2parameters(parameters), offsets, std::shared_ptr<ak::Content>(unbox_content(content)));
      }), py::arg("offsets"), py::arg("content"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("starts", &ak::ListOffsetArrayOf<T>::starts)
      .def_property_readonly("stops", &ak::ListOffsetArrayOf<T>::stops)
      .def_property_readonly("offsets", &ak::ListOffsetArrayOf<T>::offsets)
      .def_property_readonly("content", &ak::ListOffsetArrayOf<T>::content)
      .def("compact_offsets64", &ak::ListOffsetArrayOf<T>::compact_offsets64, py::arg("start_at_zero") = true)
      .def("broadcast_tooffsets64", &ak::ListOffsetArrayOf<T>::broadcast_tooffsets64)
      .def("toRegularArray", &ak::ListOffsetArrayOf<T>::toRegularArray)
      .def("simplify", [](const ak::ListOffsetArrayOf<T>& self) {
        return box(self.shallow_simplify());
      })
  );
}

template py::class_<ak::ListOffsetArray32, std::shared_ptr<ak::ListOffsetArray32>, ak::Content> make_ListOffsetArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::ListOffsetArrayU32, std::shared_ptr<ak::ListOffsetArrayU32>, ak::Content> make_ListOffsetArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::ListOffsetArray64, std::shared_ptr<ak::ListOffsetArray64>, ak::Content> make_ListOffsetArrayOf(const py::handle& m, const std::string& name);

/////////////////////////////////////////////////////////////// NumpyArray

py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content> make_NumpyArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](const ak::NumpyArray& self) -> py::buffer_info {
        return py::buffer_info(
          self.byteptr(),
          self.itemsize(),
          self.format(),
          self.ndim(),
          self.shape(),
          self.strides());
      })

      .def(py::init([](py::array& array, const py::object& identities, const py::object& parameters) -> ak::NumpyArray {
        py::buffer_info info = array.request();
        if (info.ndim == 0) {
          throw std::invalid_argument("NumpyArray must not be scalar; try array.reshape(1)");
        }
        if (info.shape.size() != info.ndim  ||  info.strides.size() != info.ndim) {
          throw std::invalid_argument("NumpyArray len(shape) != ndim or len(strides) != ndim");
        }
        return ak::NumpyArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<void>(
          reinterpret_cast<void*>(info.ptr), pyobject_deleter<void>(array.ptr())),
          info.shape,
          info.strides,
          0,
          info.itemsize,
          info.format);
      }), py::arg("array"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("shape", &ak::NumpyArray::shape)
      .def_property_readonly("strides", &ak::NumpyArray::strides)
      .def_property_readonly("itemsize", &ak::NumpyArray::itemsize)
      .def_property_readonly("format", &ak::NumpyArray::format)
      .def_property_readonly("ndim", &ak::NumpyArray::ndim)
      .def_property_readonly("isscalar", &ak::NumpyArray::isscalar)
      .def_property_readonly("isempty", &ak::NumpyArray::isempty)
      .def("toRegularArray", &ak::NumpyArray::toRegularArray)

      .def_property_readonly("iscontiguous", &ak::NumpyArray::iscontiguous)
      .def("contiguous", &ak::NumpyArray::contiguous)
      .def("simplify", [](const ak::NumpyArray& self) {
        return box(self.shallow_simplify());
      })
  );
}

/////////////////////////////////////////////////////////////// RecordArray

py::class_<ak::Record, std::shared_ptr<ak::Record>> make_Record(const py::handle& m, const std::string& name) {
  return py::class_<ak::Record, std::shared_ptr<ak::Record>>(m, name.c_str())
      .def(py::init([](const std::shared_ptr<ak::RecordArray>& array, int64_t at) -> ak::Record {
        return ak::Record(array, at);
      }), py::arg("array"), py::arg("at"))
      .def("__repr__", &repr<ak::Record>)
      .def_property_readonly("identities", [](const ak::Record& self) -> py::object { return box(self.identities()); })
      .def("__getitem__", &getitem<ak::Record>)
      .def("type", [](const ak::Record& self, const std::map<std::string, std::string>& typestrs) -> std::shared_ptr<ak::Type> {
        return self.type(typestrs);
      })
      .def_property("parameters", &getparameters<ak::Record>, &setparameters<ak::Record>)
      .def("setparameter", &setparameter<ak::Record>)
      .def("parameter", &parameter<ak::Record>)
      .def("purelist_parameter", &purelist_parameter<ak::Record>)
      .def("tojson", &tojson_string<ak::Record>, py::arg("pretty") = false, py::arg("maxdecimals") = py::none())
      .def("tojson", &tojson_file<ak::Record>, py::arg("destination"), py::arg("pretty") = false, py::arg("maxdecimals") = py::none(), py::arg("buffersize") = 65536)

      .def_property_readonly("array", [](const ak::Record& self) -> std::shared_ptr<const ak::RecordArray> { return self.array(); })
      .def_property_readonly("at", &ak::Record::at)
      .def_property_readonly("istuple", &ak::Record::istuple)
      .def_property_readonly("numfields", &ak::Record::numfields)
      .def("fieldindex", &ak::Record::fieldindex)
      .def("key", &ak::Record::key)
      .def("haskey", &ak::Record::haskey)
      .def("keys", &ak::Record::keys)
      .def("field", [](const ak::Record& self, int64_t fieldindex) -> py::object {
        return box(self.field(fieldindex));
      })
      .def("field", [](const ak::Record& self, const std::string& key) -> py::object {
        return box(self.field(key));
      })
      .def("fields", [](const ak::Record& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](const ak::Record& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def_property_readonly("astuple", [](const ak::Record& self) -> py::object {
        return box(self.astuple());
      })
     .def_property_readonly("identity", &identity<ak::Record>)
     .def("simplify", [](const ak::Record& self) {
       return box(self.shallow_simplify());
     })

  ;
}

ak::RecordArray iterable_to_RecordArray(const py::iterable& contents, const py::object& keys, const py::object& length, const py::object& identities, const py::object& parameters) {
  std::vector<std::shared_ptr<ak::Content>> out;
  for (auto x : contents) {
    out.push_back(unbox_content(x));
  }
  std::shared_ptr<ak::util::RecordLookup> recordlookup(nullptr);
  if (!keys.is(py::none())) {
    recordlookup = std::make_shared<ak::util::RecordLookup>();
    for (auto x : keys.cast<py::iterable>()) {
      recordlookup.get()->push_back(x.cast<std::string>());
    }
    if (out.size() != recordlookup.get()->size()) {
      throw std::invalid_argument("if provided, 'keys' must have the same length as 'types'");
    }
  }
  if (length.is(py::none())) {
    return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup);
  }
  else {
    int64_t intlength = length.cast<int64_t>();
    return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup, intlength);
  }
}

py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content> make_RecordArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content>(m, name.c_str())
      .def(py::init([](const py::dict& contents, const py::object& length, const py::object& identities, const py::object& parameters) -> ak::RecordArray {
        std::shared_ptr<ak::util::RecordLookup> recordlookup = std::make_shared<ak::util::RecordLookup>();
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto x : contents) {
          std::string key = x.first.cast<std::string>();
          recordlookup.get()->push_back(key);
          out.push_back(unbox_content(x.second));
        }
        if (length.is(py::none())) {
          return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup);
        }
        else {
          int64_t intlength = length.cast<int64_t>();
          return ak::RecordArray(unbox_identities_none(identities), dict2parameters(parameters), out, recordlookup, intlength);
        }
      }), py::arg("contents"), py::arg("length") = py::none(), py::arg("identities") = py::none(), py::arg("parameters") = py::none())
      .def(py::init(&iterable_to_RecordArray), py::arg("contents"), py::arg("keys") = py::none(), py::arg("length") = py::none(), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("recordlookup", [](const ak::RecordArray& self) -> py::object {
        std::shared_ptr<ak::util::RecordLookup> recordlookup = self.recordlookup();
        if (recordlookup.get() == nullptr) {
          return py::none();
        }
        else {
          py::list out;
          for (auto x : *recordlookup.get()) {
            py::str pyvalue(PyUnicode_DecodeUTF8(x.data(), x.length(), "surrogateescape"));
            out.append(pyvalue);
          }
          return out;
        }
      })
      .def_property_readonly("istuple", &ak::RecordArray::istuple)
      .def_property_readonly("contents", &ak::RecordArray::contents)
      .def("setitem_field", [](const ak::RecordArray& self, const py::object& where, const py::object& what) -> py::object {
        std::shared_ptr<ak::Content> mywhat = unbox_content(what);
        if (where.is(py::none())) {
          return box(self.setitem_field(self.numfields(), mywhat));
        }
        else {
          try {
            std::string mywhere = where.cast<std::string>();
            return box(self.setitem_field(mywhere, mywhat));
          }
          catch (py::cast_error err) {
            try {
              int64_t mywhere = where.cast<int64_t>();
              return box(self.setitem_field(mywhere, mywhat));
            }
            catch (py::cast_error err) {
              throw std::invalid_argument("where must be None, int, or str");
            }
          }
        }
      }, py::arg("where"), py::arg("what"))

      .def("field", [](const ak::RecordArray& self, int64_t fieldindex) -> std::shared_ptr<ak::Content> {
        return self.field(fieldindex);
      })
      .def("field", [](const ak::RecordArray& self, const std::string& key) -> std::shared_ptr<ak::Content> {
        return self.field(key);
      })
      .def("fields", [](const ak::RecordArray& self) -> py::object {
        py::list out;
        for (auto item : self.fields()) {
          out.append(box(item));
        }
        return out;
      })
      .def("fielditems", [](const ak::RecordArray& self) -> py::object {
        py::list out;
        for (auto item : self.fielditems()) {
          py::str key(item.first);
          py::object val(box(item.second));
          py::tuple pair(2);
          pair[0] = key;
          pair[1] = val;
          out.append(pair);
        }
        return out;
      })
      .def_property_readonly("astuple", [](const ak::RecordArray& self) -> py::object {
        return box(self.astuple());
      })
      .def("simplify", [](const ak::RecordArray& self) {
        return box(self.shallow_simplify());
      })

  );
}

/////////////////////////////////////////////////////////////// RegularArray

py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content> make_RegularArray(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content>(m, name.c_str())
      .def(py::init([](const py::object& content, int64_t size, const py::object& identities, const py::object& parameters) -> ak::RegularArray {
        return ak::RegularArray(unbox_identities_none(identities), dict2parameters(parameters), std::shared_ptr<ak::Content>(unbox_content(content)), size);
      }), py::arg("content"), py::arg("size"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_property_readonly("size", &ak::RegularArray::size)
      .def_property_readonly("content", &ak::RegularArray::content)
      .def("compact_offsets64", &ak::RegularArray::compact_offsets64, py::arg("start_at_zero") = true)
      .def("broadcast_tooffsets64", &ak::RegularArray::broadcast_tooffsets64)
      .def("simplify", [](const ak::RegularArray& self) {
        return box(self.shallow_simplify());
      })
  );
}

/////////////////////////////////////////////////////////////// UnionArray

template <typename T, typename I>
py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content> make_UnionArrayOf(const py::handle& m, const std::string& name) {
  return content_methods(py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content>(m, name.c_str())
      .def(py::init([](const ak::IndexOf<T>& tags, ak::IndexOf<I>& index, const py::iterable& contents, const py::object& identities, const py::object& parameters) -> ak::UnionArrayOf<T, I> {
        std::vector<std::shared_ptr<ak::Content>> out;
        for (auto content : contents) {
          out.push_back(std::shared_ptr<ak::Content>(unbox_content(content)));
        }
        return ak::UnionArrayOf<T, I>(unbox_identities_none(identities), dict2parameters(parameters), tags, index, out);
      }), py::arg("tags"), py::arg("index"), py::arg("contents"), py::arg("identities") = py::none(), py::arg("parameters") = py::none())

      .def_static("regular_index", &ak::UnionArrayOf<T, I>::regular_index)
      .def_property_readonly("tags", &ak::UnionArrayOf<T, I>::tags)
      .def_property_readonly("index", &ak::UnionArrayOf<T, I>::index)
      .def_property_readonly("contents", &ak::UnionArrayOf<T, I>::contents)
      .def_property_readonly("numcontents", &ak::UnionArrayOf<T, I>::numcontents)
      .def("content", &ak::UnionArrayOf<T, I>::content)
      .def("project", &ak::UnionArrayOf<T, I>::project)
      .def("simplify", [](const ak::UnionArrayOf<T, I>& self, bool mergebool) -> py::object {
        return box(self.simplify_uniontype(mergebool));
      }, py::arg("mergebool") = false)

  );
}

template py::class_<ak::UnionArray8_32, std::shared_ptr<ak::UnionArray8_32>, ak::Content> make_UnionArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::UnionArray8_U32, std::shared_ptr<ak::UnionArray8_U32>, ak::Content> make_UnionArrayOf(const py::handle& m, const std::string& name);
template py::class_<ak::UnionArray8_64, std::shared_ptr<ak::UnionArray8_64>, ak::Content> make_UnionArrayOf(const py::handle& m, const std::string& name);
