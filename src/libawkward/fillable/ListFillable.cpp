// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/ListType.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/ListFillable.h"

namespace awkward {
  int64_t ListFillable::length() const {
    return offsets_.length() - 1;
  }

  void ListFillable::clear() {
    offsets_.clear();
    content_.get()->clear();
  }

  const std::shared_ptr<Type> ListFillable::type() const {
    return std::shared_ptr<Type>(new ListType(content_.get()->type()));
  }

  const std::shared_ptr<Content> ListFillable::snapshot() const {
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    return std::shared_ptr<Content>(new ListOffsetArray64(Identity::none(), offsets, content_.get()->snapshot()));
  }

  bool ListFillable::active() const {
    throw std::runtime_error("FIXME: ListFillable::active");
  }

  Fillable* ListFillable::null() {
    throw std::runtime_error("FIXME: ListFillable::null");
  }

  Fillable* ListFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: ListFillable::boolean");
  }

  Fillable* ListFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: ListFillable::integer");
  }

  Fillable* ListFillable::real(double x) {
    throw std::runtime_error("FIXME: ListFillable::real");
  }

  Fillable* ListFillable::beginlist() {
    throw std::runtime_error("FIXME: ListFillable::beginlist");
  }

  Fillable* ListFillable::endlist() {
    throw std::runtime_error("FIXME: ListFillable::endlist");
  }

  Fillable* ListFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: ListFillable::begintuple");
  }

  Fillable* ListFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: ListFillable::index");
  }

  Fillable* ListFillable::endtuple() {
    throw std::runtime_error("FIXME: ListFillable::endtuple");
  }

  Fillable* ListFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: ListFillable::beginrecord");
  }

  Fillable* ListFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: ListFillable::field_fast");
  }

  Fillable* ListFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: ListFillable::field_check");
  }

  Fillable* ListFillable::endrecord() {
    throw std::runtime_error("FIXME: ListFillable::endrecord");
  }

  Fillable* ListFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()  &&  tmp != nullptr) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
    return this;
  }
}
