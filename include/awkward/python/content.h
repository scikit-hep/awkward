// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_CONTENT_H_
#define AWKWARDPY_CONTENT_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/Iterator.h"
#include "awkward/Content.h"
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

namespace py = pybind11;
namespace ak = awkward;

ak::Slice toslice(py::object obj);

py::class_<ak::ArrayBuilder> make_ArrayBuilder(const py::handle& m, const std::string& name);

py::class_<ak::Iterator, std::shared_ptr<ak::Iterator>> make_Iterator(const py::handle& m, const std::string& name);

class PersistentSharedPtr {
public:
  PersistentSharedPtr(const std::shared_ptr<ak::Content>& ptr);
  py::object layout() const;
  size_t ptr() const;

private:
  const std::shared_ptr<ak::Content> ptr_;
};

py::class_<PersistentSharedPtr> make_PersistentSharedPtr(const py::handle& m, const std::string& name);

py::class_<ak::Content, std::shared_ptr<ak::Content>> make_Content(const py::handle& m, const std::string& name);

py::class_<ak::EmptyArray, std::shared_ptr<ak::EmptyArray>, ak::Content> make_EmptyArray(const py::handle& m, const std::string& name);

template <typename T, bool ISOPTION>
py::class_<ak::IndexedArrayOf<T, ISOPTION>, std::shared_ptr<ak::IndexedArrayOf<T, ISOPTION>>, ak::Content> make_IndexedArrayOf(const py::handle& m, const std::string& name);

template <typename T>
py::class_<ak::ListArrayOf<T>, std::shared_ptr<ak::ListArrayOf<T>>, ak::Content> make_ListArrayOf(const py::handle& m, const std::string& name);

template <typename T>
py::class_<ak::ListOffsetArrayOf<T>, std::shared_ptr<ak::ListOffsetArrayOf<T>>, ak::Content> make_ListOffsetArrayOf(const py::handle& m, const std::string& name);

py::class_<ak::NumpyArray, std::shared_ptr<ak::NumpyArray>, ak::Content> make_NumpyArray(const py::handle& m, const std::string& name);

py::class_<ak::Record, std::shared_ptr<ak::Record>> make_Record(const py::handle& m, const std::string& name);

py::class_<ak::RecordArray, std::shared_ptr<ak::RecordArray>, ak::Content> make_RecordArray(const py::handle& m, const std::string& name);

py::class_<ak::RegularArray, std::shared_ptr<ak::RegularArray>, ak::Content> make_RegularArray(const py::handle& m, const std::string& name);

template <typename T, typename I>
py::class_<ak::UnionArrayOf<T, I>, std::shared_ptr<ak::UnionArrayOf<T, I>>, ak::Content> make_UnionArrayOf(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_CONTENT_H_
