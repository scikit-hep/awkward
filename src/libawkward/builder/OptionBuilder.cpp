// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/type/OptionType.h"

#include "awkward/builder/OptionBuilder.h"

namespace awkward {
  const std::shared_ptr<Builder> OptionBuilder::fromnulls(const ArrayBuilderOptions& options, int64_t nullcount, const std::shared_ptr<Builder>& content) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::full(options, -1, nullcount);
    std::shared_ptr<Builder> out = std::make_shared<OptionBuilder>(options, offsets, content);
    out.get()->setthat(out);
    return out;
  }

  const std::shared_ptr<Builder> OptionBuilder::fromvalids(const ArrayBuilderOptions& options, const std::shared_ptr<Builder>& content) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::arange(options, content->length());
    std::shared_ptr<Builder> out = std::make_shared<OptionBuilder>(options, offsets, content);
    out.get()->setthat(out);
    return out;
  }

  OptionBuilder::OptionBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& offsets, const std::shared_ptr<Builder>& content)
      : options_(options)
      , offsets_(offsets)
      , content_(content) { }

  const std::string OptionBuilder::classname() const {
    return "OptionBuilder";
  };

  int64_t OptionBuilder::length() const {
    return offsets_.length();
  }

  void OptionBuilder::clear() {
    offsets_.clear();
    content_.get()->clear();
  }

  ContentPtr OptionBuilder::snapshot() const {
    Index64 index(offsets_.ptr(), 0, offsets_.length());
    return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, content_.get()->snapshot());
  }

  bool OptionBuilder::active() const {
    return content_.get()->active();
  }

  const std::shared_ptr<Builder> OptionBuilder::null() {
    if (!content_.get()->active()) {
      offsets_.append(-1);
    }
    else {
      content_.get()->null();
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::boolean(bool x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->boolean(x));
      offsets_.append(length);
    }
    else {
      content_.get()->boolean(x);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::integer(int64_t x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->integer(x));
      offsets_.append(length);
    }
    else {
      content_.get()->integer(x);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::real(double x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->real(x));
      offsets_.append(length);
    }
    else {
      content_.get()->real(x);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!content_.get()->active()) {
      int64_t len = content_.get()->length();
      maybeupdate(content_.get()->string(x, length, encoding));
      offsets_.append(len);
    }
    else {
      content_.get()->string(x, length, encoding);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::beginlist() {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->beginlist());
    }
    else {
      content_.get()->beginlist();
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::endlist() {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endlist();
      if (length != content_.get()->length()) {
        offsets_.append(length);
      }
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::begintuple(int64_t numfields) {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->begintuple(numfields));
    }
    else {
      content_.get()->begintuple(numfields);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::index(int64_t index) {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->index(index);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::endtuple() {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endtuple();
      if (length != content_.get()->length()) {
        offsets_.append(length);
      }
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::beginrecord(const char* name, bool check) {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->beginrecord(name, check));
    }
    else {
      content_.get()->beginrecord(name, check);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::field(const char* key, bool check) {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field(key, check);
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::endrecord() {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endrecord();
      if (length != content_.get()->length()) {
        offsets_.append(length);
      }
    }
    return that_;
  }

  const std::shared_ptr<Builder> OptionBuilder::append(ContentPtr& array, int64_t at) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->append(array, at));
      offsets_.append(length);
    }
    else {
      content_.get()->append(array, at);
    }
    return that_;
  }

  void OptionBuilder::maybeupdate(const std::shared_ptr<Builder>& tmp) {
    if (tmp.get() != content_.get()) {
      content_ = tmp;
    }
  }
}
