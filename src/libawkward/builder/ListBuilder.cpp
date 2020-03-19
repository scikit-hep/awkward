// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/ListType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/ListBuilder.h"

namespace awkward {
  const std::shared_ptr<Builder> ListBuilder::fromempty(const ArrayBuilderOptions& options) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    std::shared_ptr<Builder> out = std::make_shared<ListBuilder>(options, offsets, UnknownBuilder::fromempty(options), false);
    out.get()->setthat(out);
    return out;
  }

  ListBuilder::ListBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& offsets, const std::shared_ptr<Builder>& content, bool begun)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , begun_(begun) { }

  const std::string ListBuilder::classname() const {
    return "ListBuilder";
  };

  int64_t ListBuilder::length() const {
    return offsets_.length() - 1;
  }

  void ListBuilder::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.get()->clear();
  }

  ContentPtr ListBuilder::snapshot() const {
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    return std::make_shared<ListOffsetArray64>(Identities::none(), util::Parameters(), offsets, content_.get()->snapshot());
  }

  bool ListBuilder::active() const {
    return begun_;
  }

  const std::shared_ptr<Builder> ListBuilder::null() {
    if (!begun_) {
      std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else {
      maybeupdate(content_.get()->null());
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::boolean(bool x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->boolean(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->boolean(x));
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::integer(int64_t x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->integer(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->integer(x));
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::real(double x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->real(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->real(x));
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->string(x, length, encoding);
      return out;
    }
    else {
      maybeupdate(content_.get()->string(x, length, encoding));
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::beginlist() {
    if (!begun_) {
      begun_ = true;
    }
    else {
      maybeupdate(content_.get()->beginlist());
    }
    return that_;
  }

  const std::shared_ptr<Builder> ListBuilder::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (!content_.get()->active()) {
      offsets_.append(content_.get()->length());
      begun_ = false;
    }
    else {
      maybeupdate(content_.get()->endlist());
    }
    return that_;
  }

  const std::shared_ptr<Builder> ListBuilder::begintuple(int64_t numfields) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->begintuple(numfields);
      return out;
    }
    else {
      maybeupdate(content_.get()->begintuple(numfields));
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->index(index);
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->endtuple();
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::beginrecord(const char* name, bool check) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->beginrecord(name, check);
      return out;
    }
    else {
      maybeupdate(content_.get()->beginrecord(name, check));
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field(key, check);
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->endrecord();
      return that_;
    }
  }

  const std::shared_ptr<Builder> ListBuilder::append(ContentPtr& array, int64_t at) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    else {
      maybeupdate(content_.get()->append(array, at));
      return that_;
    }
  }

  void ListBuilder::maybeupdate(const std::shared_ptr<Builder>& tmp) {
    if (tmp.get() != content_.get()) {
      content_ = tmp;
    }
  }
}
