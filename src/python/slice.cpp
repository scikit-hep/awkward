// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>

#include "awkward/Content.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"

#include "awkward/python/boxing.h"
#include "awkward/python/util.h"

#include "awkward/python/slice.h"

bool handle_as_numpy(const std::shared_ptr<ak::Content>& content) {
  // if (content.get()->parameter_equals("__array__", "\"string\"")) {
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
      else if (py::isinstance<ak::FillableArray>(obj)) {
        content = unbox_content(obj.attr("snapshot")());
      }
      else if (py::isinstance(obj, py::module::import("awkward1").attr("Array"))) {
        content = unbox_content(obj.attr("layout"));
      }
      else if (py::isinstance(obj, py::module::import("awkward1").attr("FillableArray"))) {
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
        if (content.get()->parameter_equals("__array__", "\"string\"")) {
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

void make_slice_tostring(py::module& m, const std::string& name) {
  m.def(name.c_str(), [](py::object obj) -> std::string {
    return toslice(obj).tostring();
  });
}
