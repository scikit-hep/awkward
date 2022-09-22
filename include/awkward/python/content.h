// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARDPY_CONTENT_H_
#define AWKWARDPY_CONTENT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/util.h"

namespace py = pybind11;
namespace ak = awkward;

template <typename T>
std::string
repr(const T& self) {
  return self.tostring();
}

template <typename T>
int64_t
len(const T& self) {
  return self.length();
}

int64_t
check_maxdecimals(const py::object& maxdecimals);

ak::util::Parameters
dict2parameters(const py::object& in);

py::dict
parameters2dict(const ak::util::Parameters& in);

/// @brief Makes an ArrayBuilder class in Python that mirrors the one in C++.
py::class_<ak::ArrayBuilder>
  make_ArrayBuilder(const py::handle& m, const std::string& name);

namespace {
  class NumpyBuffersContainer: public ak::BuffersContainer {
  public:
    py::dict container() {
      return container_;
    }

    void*
      empty_buffer(const std::string& name, int64_t num_bytes) override {
        py::object pyarray = py::module::import("numpy").attr("empty")(num_bytes, "u1");
        py::array_t<uint8_t> rawarray = pyarray.cast<py::array_t<uint8_t>>();
        py::buffer_info rawinfo = rawarray.request();
        container_[py::str(name)] = pyarray;
        return rawinfo.ptr;
      }

    void
      copy_buffer(const std::string& name, const void* source, int64_t num_bytes) override {
        py::object pyarray = py::module::import("numpy").attr("empty")(num_bytes, "u1");
        py::array_t<uint8_t> rawarray = pyarray.cast<py::array_t<uint8_t>>();
        py::buffer_info rawinfo = rawarray.request();
        std::memcpy(rawinfo.ptr, source, num_bytes);
        container_[py::str(name)] = pyarray;
      }

    void
      full_buffer(const std::string& name, int64_t length, int64_t value, const std::string& dtype) override {
        py::object pyarray = py::module::import("numpy").attr("full")(py::int_(length), py::int_(value), py::str(dtype));
        container_[py::str(name)] = pyarray;
      }

  private:
    py::dict container_;
  };

  class EmptyBuffersContainer: public ak::BuffersContainer {
  public:
    void*
      empty_buffer(const std::string& name, int64_t num_bytes) override {
        return nullptr;
      }

    void
      copy_buffer(const std::string& name, const void* source, int64_t num_bytes) override { }

    void
      full_buffer(const std::string& name, int64_t length, int64_t value, const std::string& dtype) override { }
  };

  /// @brief Turns the accumulated data into a Content array.
  ///
  /// This operation only converts Builder nodes into Content nodes; the
  /// buffers holding array data are shared between the Builder and the
  /// Content. Hence, taking a snapshot is a constant-time operation.
  ///
  /// It is safe to take multiple snapshots while accumulating data. The
  /// shared buffers are only appended to, which affects elements beyond
  /// the limited view of old snapshots.
  py::object
  builder_snapshot(const ak::BuilderPtr builder) {
    ::NumpyBuffersContainer container;
    int64_t form_key_id = 0;
    std::string form = builder.get()->to_buffers(container, form_key_id);
    py::dict kwargs;
    kwargs[py::str("form")] = py::str(form);
    kwargs[py::str("length")] = py::int_(builder.get()->length());
    kwargs[py::str("container")] = container.container();
    kwargs[py::str("key_format")] = py::str("{form_key}-{attribute}");
    kwargs[py::str("highlevel")] = py::bool_(false);
    return py::module::import("awkward").attr("from_buffers")(**kwargs);
  }
}

#endif // AWKWARDPY_CONTENT_H_
