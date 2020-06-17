// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_CONTENT_H_
#define AWKWARDPY_CONTENT_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/Iterator.h"
#include "awkward/Content.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/None.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/Record.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/VirtualArray.h"

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

/// @brief Makes an Iterator class in Python that mirrors the one in C++.
py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>>
  make_Iterator(const py::handle& m, const std::string& name);

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

/// @brief Makes an EmptyArray in Python that mirrors the one in C++.
py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content>
  make_EmptyArray(const py::handle& m, const std::string& name);

/// @brief Makes IndexedArray32, IndexedArrayU32, IndexedArray64,
/// IndexedOptionArray32, and IndexedOptionArray64 classes in Python that
/// mirror IndexedArrayOf in C++.
template <typename T, bool ISOPTION>
py::class_<ak::IndexedArrayOf<T, ISOPTION>,
           std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>,
           ak::Content>
  make_IndexedArrayOf(const py::handle& m, const std::string& name);

/// @brief Makes a ByteMaskedArray in Python that mirrors the one in C++.
py::class_<ak::ByteMaskedArray,
           std::shared_ptr<ak::ByteMaskedArray>,
           ak::Content>
  make_ByteMaskedArray(const py::handle& m, const std::string& name);

/// @brief Makes a BitMaskedArray in Python that mirrors the one in C++.
py::class_<ak::BitMaskedArray,
           std::shared_ptr<ak::BitMaskedArray>,
           ak::Content>
  make_BitMaskedArray(const py::handle& m, const std::string& name);

/// @brief Makes a UnmaskedArray in Python that mirrors the one in C++.
py::class_<ak::UnmaskedArray,
           std::shared_ptr<ak::UnmaskedArray>,
           ak::Content>
  make_UnmaskedArray(const py::handle& m, const std::string& name);

/// @brief Makes ListArray32, ListArrayU32, and ListArray64 classes in
/// Python that mirror ListArrayOf in C++.
template <typename T>
py::class_<ak::ListArrayOf<T>,
           std::shared_ptr<ak::ListArrayOf<T>>,
           ak::Content>
  make_ListArrayOf(const py::handle& m, const std::string& name);

/// @brief Makes ListOffsetArray32, ListOffsetArrayU32, and ListOffsetArray64
/// classes in Python that mirror ListOffsetArrayOf in C++.
template <typename T>
py::class_<ak::ListOffsetArrayOf<T>,
           std::shared_ptr<ak::ListOffsetArrayOf<T>>,
           ak::Content>
  make_ListOffsetArrayOf(const py::handle& m, const std::string& name);

/// @brief Makes a NumpyArray in Python that mirrors the one in C++.
py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content>
  make_NumpyArray(const py::handle& m, const std::string& name);

/// @brief Makes a Record in Python that mirrors the one in C++.
///
/// Note that the Python Record class does not inherit from the Python
/// Content class, although Record inherits from Content in C++. Ideally,
/// we'd prefer scalars to not have the same type as arrays, but that's
/// easier to do in Python.
py::class_<ak::Record, std::shared_ptr<ak::Record>>
  make_Record(const py::handle& m, const std::string& name);

/// @brief Makes a RecordArray in Python that mirrors the one in C++.
py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content>
  make_RecordArray(const py::handle& m, const std::string& name);

/// @brief Makes a RegularArray in Python that mirrors the one in C++.
py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content>
  make_RegularArray(const py::handle& m, const std::string& name);

/// @brief Makes UnionArray8_32, UnionArray8_U32, and UnionArray8_64 classes
/// in Python that mirror UnionArrayOf in C++.
template <typename T, typename I>
py::class_<ak::UnionArrayOf<T, I>,
           std::shared_ptr<ak::UnionArrayOf<T, I>>,
           ak::Content>
  make_UnionArrayOf(const py::handle& m, const std::string& name);

/// @brief Makes a VirtualArray in Python that mirrors the one in C++.
py::class_<ak::VirtualArray, std::shared_ptr<ak::VirtualArray>, ak::Content>
  make_VirtualArray(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_CONTENT_H_
