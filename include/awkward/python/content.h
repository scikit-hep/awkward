// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARDPY_CONTENT_H_
#define AWKWARDPY_CONTENT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"
#include "awkward/Content.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"

namespace py = pybind11;
namespace ak = awkward;

py::object
box(const std::shared_ptr<ak::Content>& content);

std::shared_ptr<ak::Content>
unbox_content(const py::handle& obj);

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

/// @brief Converts Python objects in a slice into a C++ Slice.
ak::Slice
  toslice(py::object obj);

/// @brief Makes an ArrayBuilder class in Python that mirrors the one in C++.
py::class_<ak::ArrayBuilder>
  make_ArrayBuilder(const py::handle& m, const std::string& name);

/// @brief Makes a LayoutBuilder class in Python that mirrors the one in C++.
template <typename T, typename I>
py::class_<ak::LayoutBuilder<T, I>>
  make_LayoutBuilder(const py::handle& m, const std::string& name);

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

/// @class PersistentSharedPtr
///
/// @brief Array nodes are frequently copied, but for some applications
/// (one in Numba) it's better to keep a persistent `std::shared_ptr`.
class PersistentSharedPtr {
public:
  /// @brief Creates a PersistentSharedPtr from a `std::shared_ptr` to a
  /// Content array.
  PersistentSharedPtr(const std::shared_ptr<ak::Content>& ptr);

  /// @brief Returns a layout object (Content in Python) from this #ptr.
  py::object
    layout() const;

  /// @brief Returns a raw pointer to the persistent `std::shared_ptr`
  /// (a raw pointer to a smart pointer).
  size_t
    ptr() const;

private:
  /// @brief The wrapped `std::shared_ptr`.
  const std::shared_ptr<ak::Content> ptr_;
};

/// @brief Makes a PersistentSharedPtr class in Python that mirrors the one
/// in C++.
py::class_<PersistentSharedPtr>
  make_PersistentSharedPtr(const py::handle& m, const std::string& name);

/// @brief Makes an abstract Content class in Python that mirrors the one
/// in C++.
py::class_<ak::Content, std::shared_ptr<ak::Content>>
  make_Content(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_CONTENT_H_
