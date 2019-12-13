// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"
#include "awkward/fillable/Int64Fillable.h"
#include "awkward/fillable/Float64Fillable.h"
#include "awkward/fillable/StringFillable.h"
#include "awkward/fillable/ListFillable.h"
#include "awkward/fillable/TupleFillable.h"
#include "awkward/fillable/RecordFillable.h"

#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> UnknownFillable::fromempty(const FillableOptions& options) {
    std::shared_ptr<Fillable> out(new UnknownFillable(options, 0));
    out.get()->setthat(out);
    return out;
  }

  int64_t UnknownFillable::length() const {
    return nullcount_;
  }

  void UnknownFillable::clear() {
    nullcount_ = 0;
  }

  const std::shared_ptr<Type> UnknownFillable::type() const {
    return std::shared_ptr<Type>(new UnknownType(Type::Parameters()));
  }

  const std::shared_ptr<Content> UnknownFillable::snapshot() const {
    if (nullcount_ == 0) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none(), Type::none()));
    }
    else {
      throw std::runtime_error("UnknownFillable::snapshot() needs OptionArray");
    }
  }

  bool UnknownFillable::active() const {
    return false;
  }

  const std::shared_ptr<Fillable> UnknownFillable::null() {
    nullcount_++;
    return that_;
  }

  const std::shared_ptr<Fillable> UnknownFillable::boolean(bool x) {
    std::shared_ptr<Fillable> out = BoolFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::integer(int64_t x) {
    std::shared_ptr<Fillable> out = Int64Fillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::real(double x) {
    std::shared_ptr<Fillable> out = Float64Fillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Fillable> out = StringFillable::fromempty(options_, encoding);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::beginlist() {
    std::shared_ptr<Fillable> out = ListFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }

    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Fillable> UnknownFillable::begintuple(int64_t numfields) {
    std::shared_ptr<Fillable> out = TupleFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> UnknownFillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> UnknownFillable::beginrecord(const char* name, bool check) {
    std::shared_ptr<Fillable> out = RecordFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Fillable> UnknownFillable::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Fillable> UnknownFillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }
}
