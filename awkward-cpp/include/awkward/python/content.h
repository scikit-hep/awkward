// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARDPY_CONTENT_H_
#define AWKWARDPY_CONTENT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/util.h"

namespace py = pybind11;
namespace ak = awkward;


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
      empty_buffer(const std::string& /* name */, int64_t /* num_bytes */) override {
        return nullptr;
      }

    void
      copy_buffer(const std::string& /* name */, const void* /* source */, int64_t /* num_bytes */) override { }

    void
      full_buffer(const std::string& /* name */, int64_t /* length */, int64_t /* value */, const std::string& /* dtype */) override { }
  };
}

#endif // AWKWARDPY_CONTENT_H_
