// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
#include "awkward/type/ArrayType.h"
#include "awkward/type/ListType.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/UnionType.h"
#include "awkward/type/UnknownType.h"

#include "awkward/python/boxing.h"

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

py::object box(const std::shared_ptr<ak::Type>& t) {
  if (ak::ArrayType* raw = dynamic_cast<ak::ArrayType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::ListType* raw = dynamic_cast<ak::ListType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::OptionType* raw = dynamic_cast<ak::OptionType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::PrimitiveType* raw = dynamic_cast<ak::PrimitiveType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::RecordType* raw = dynamic_cast<ak::RecordType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::RegularType* raw = dynamic_cast<ak::RegularType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnionType* raw = dynamic_cast<ak::UnionType*>(t.get())) {
    return py::cast(*raw);
  }
  else if (ak::UnknownType* raw = dynamic_cast<ak::UnknownType*>(t.get())) {
    return py::cast(*raw);
  }
  else {
    throw std::runtime_error("missing boxer for Type subtype");
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

std::shared_ptr<ak::Identities> unbox_identities_none(const py::handle& obj) {
  if (obj.is(py::none())) {
    return ak::Identities::none();
  }
  else {
    return unbox_identities(obj);
  }
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

std::shared_ptr<ak::Type> unbox_type(const py::handle& obj) {
  try {
    return obj.cast<ak::ArrayType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::ListType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::OptionType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::PrimitiveType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RecordType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::RegularType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnionType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  try {
    return obj.cast<ak::UnknownType*>()->shallow_copy();
  }
  catch (py::cast_error err) { }
  throw std::invalid_argument("argument must be a Type subtype");
}
