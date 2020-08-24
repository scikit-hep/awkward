// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/UnknownBuilder.cpp", line)

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/BoolBuilder.h"
#include "awkward/builder/Int64Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/StringBuilder.h"
#include "awkward/builder/ListBuilder.h"
#include "awkward/builder/TupleBuilder.h"
#include "awkward/builder/RecordBuilder.h"
#include "awkward/builder/IndexedBuilder.h"

#include "awkward/builder/UnknownBuilder.h"

namespace awkward {
  const BuilderPtr
  UnknownBuilder::fromempty(const ArrayBuilderOptions& options) {
    BuilderPtr out = std::make_shared<UnknownBuilder>(options, 0);
    out.get()->setthat(out);
    return out;
  }

  UnknownBuilder::UnknownBuilder(const ArrayBuilderOptions& options,
                                 int64_t nullcount)
      : options_(options)
      , nullcount_(nullcount) { }

  const std::string
  UnknownBuilder::classname() const {
    return "UnknownBuilder";
  };

  int64_t
  UnknownBuilder::length() const {
    return nullcount_;
  }

  void
  UnknownBuilder::clear() {
    nullcount_ = 0;
  }

  const ContentPtr
  UnknownBuilder::snapshot() const {
    if (nullcount_ == 0) {
      return std::make_shared<EmptyArray>(Identities::none(),
                                          util::Parameters());
    }
    else {
      // This is the only snapshot that is O(N), rather than O(1),
      // but it is a corner case (array of only Nones).
      Index64 index(nullcount_);
      int64_t* rawptr = index.ptr().get();
      for (int64_t i = 0;  i < nullcount_;  i++) {
        rawptr[i] = -1;
      }
      return std::make_shared<IndexedOptionArray64>(
        Identities::none(),
        util::Parameters(),
        index,
        std::make_shared<EmptyArray>(Identities::none(), util::Parameters()));
    }
  }

  bool
  UnknownBuilder::active() const {
    return false;
  }

  const BuilderPtr
  UnknownBuilder::null() {
    nullcount_++;
    return that_;
  }

  const BuilderPtr
  UnknownBuilder::boolean(bool x) {
    BuilderPtr out = BoolBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  UnknownBuilder::integer(int64_t x) {
    BuilderPtr out = Int64Builder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  UnknownBuilder::real(double x) {
    BuilderPtr out = Float64Builder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  UnknownBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = StringBuilder::fromempty(options_, encoding);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  UnknownBuilder::beginlist() {
    BuilderPtr out = ListBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  UnknownBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = TupleBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  UnknownBuilder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = RecordBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  UnknownBuilder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = IndexedGenericBuilder::fromnulls(options_,
                                                      nullcount_,
                                                      array);
    out.get()->append(array, at);
    return out;
  }
}
