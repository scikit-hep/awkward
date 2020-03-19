// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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
  const std::shared_ptr<Builder> UnknownBuilder::fromempty(const ArrayBuilderOptions& options) {
    std::shared_ptr<Builder> out = std::make_shared<UnknownBuilder>(options, 0);
    out.get()->setthat(out);
    return out;
  }

  UnknownBuilder::UnknownBuilder(const ArrayBuilderOptions& options, int64_t nullcount)
      : options_(options)
      , nullcount_(nullcount) { }

  const std::string UnknownBuilder::classname() const {
    return "UnknownBuilder";
  };

  int64_t UnknownBuilder::length() const {
    return nullcount_;
  }

  void UnknownBuilder::clear() {
    nullcount_ = 0;
  }

  ContentPtr UnknownBuilder::snapshot() const {
    if (nullcount_ == 0) {
      return std::make_shared<EmptyArray>(Identities::none(), util::Parameters());
    }
    else {
      // This is the only snapshot that is O(N), rather than O(1), but it is unusual (array of only None).
      Index64 index(nullcount_);
      int64_t* rawptr = index.ptr().get();
      for (int64_t i = 0;  i < nullcount_;  i++) {
        rawptr[i] = -1;
      }
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, std::make_shared<EmptyArray>(Identities::none(), util::Parameters()));
    }
  }

  bool UnknownBuilder::active() const {
    return false;
  }

  const std::shared_ptr<Builder> UnknownBuilder::null() {
    nullcount_++;
    return that_;
  }

  const std::shared_ptr<Builder> UnknownBuilder::boolean(bool x) {
    std::shared_ptr<Builder> out = BoolBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::integer(int64_t x) {
    std::shared_ptr<Builder> out = Int64Builder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::real(double x) {
    std::shared_ptr<Builder> out = Float64Builder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Builder> out = StringBuilder::fromempty(options_, encoding);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::beginlist() {
    std::shared_ptr<Builder> out = ListBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Builder> UnknownBuilder::begintuple(int64_t numfields) {
    std::shared_ptr<Builder> out = TupleBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> UnknownBuilder::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> UnknownBuilder::beginrecord(const char* name, bool check) {
    std::shared_ptr<Builder> out = RecordBuilder::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(options_, nullcount_, out);
    }
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Builder> UnknownBuilder::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> UnknownBuilder::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> UnknownBuilder::append(ContentPtr& array, int64_t at) {
    std::shared_ptr<Builder> out = IndexedGenericBuilder::fromnulls(options_, nullcount_, array);
    out.get()->append(array, at);
    return out;
  }
}
