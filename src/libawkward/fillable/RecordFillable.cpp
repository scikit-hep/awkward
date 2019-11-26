// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/ListType.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/RecordFillable.h"

namespace awkward {
  int64_t RecordFillable::length() const {
    throw std::runtime_error("FIXME: RecordFillable::length");
  }

  void RecordFillable::clear() {
    throw std::runtime_error("FIXME: RecordFillable::clear");
  }

  const std::shared_ptr<Type> RecordFillable::type() const {
    throw std::runtime_error("FIXME: RecordFillable::type");
  }

  const std::shared_ptr<Content> RecordFillable::snapshot() const {
    throw std::runtime_error("FIXME: RecordFillable::snapshot");
  }

  Fillable* RecordFillable::null() {
    throw std::runtime_error("FIXME: RecordFillable::null");
  }

  Fillable* RecordFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: RecordFillable::boolean");
  }

  Fillable* RecordFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: RecordFillable::integer");
  }

  Fillable* RecordFillable::real(double x) {
    throw std::runtime_error("FIXME: RecordFillable::real");
  }

  Fillable* RecordFillable::beginlist() {
    throw std::runtime_error("FIXME: RecordFillable::beginlist");
  }

  Fillable* RecordFillable::endlist() {
    throw std::runtime_error("FIXME: RecordFillable::endlist");
  }

  Fillable* RecordFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: RecordFillable::begintuple");
  }

  Fillable* RecordFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: RecordFillable::index(int)");
  }

  Fillable* RecordFillable::endtuple() {
    throw std::runtime_error("FIXME: RecordFillable::endtuple");
  }

}
