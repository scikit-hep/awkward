// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/TupleBuilder.h"

namespace awkward {
  const std::shared_ptr<Builder> TupleBuilder::fromempty(const ArrayBuilderOptions& options) {
    std::shared_ptr<Builder> out = std::make_shared<TupleBuilder>(options, std::vector<std::shared_ptr<Builder>>(), -1, false, -1);
    out.get()->setthat(out);
    return out;
  }

  TupleBuilder::TupleBuilder(const ArrayBuilderOptions& options, const std::vector<std::shared_ptr<Builder>>& contents, int64_t length, bool begun, size_t nextindex)
      : options_(options)
      , contents_(contents)
      , length_(length)
      , begun_(begun)
      , nextindex_(nextindex) { }

  int64_t TupleBuilder::numfields() const {
    return (int64_t)contents_.size();
  }

  const std::string TupleBuilder::classname() const {
    return "TupleBuilder";
  };

  int64_t TupleBuilder::length() const {
    return length_;
  }

  void TupleBuilder::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    length_ = -1;
    begun_ = false;
    nextindex_ = -1;
  }

  const std::shared_ptr<Content> TupleBuilder::snapshot() const {
    if (length_ == -1) {
      return std::make_shared<EmptyArray>(Identities::none(), util::Parameters());
    }
    std::vector<std::shared_ptr<Content>> contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      contents.push_back(contents_[i].get()->snapshot());
    }
    if (contents.empty()) {
      return std::make_shared<RecordArray>(Identities::none(), util::Parameters(), length_, true);
    }
    else {
      return std::make_shared<RecordArray>(Identities::none(), util::Parameters(), contents);
    }
  }

  bool TupleBuilder::active() const {
    return begun_;
  }

  const std::shared_ptr<Builder> TupleBuilder::null() {
    if (!begun_) {
      std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'null' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::boolean(bool x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->boolean(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'boolean' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::integer(int64_t x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->integer(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'integer' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::real(double x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->real(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'real' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->string(x, length, encoding);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'string' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->string(x, length, encoding));
    }
    else {
      contents_[(size_t)nextindex_].get()->string(x, length, encoding);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::beginlist() {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->beginlist();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginlist' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginlist());
    }
    else {
      contents_[(size_t)nextindex_].get()->beginlist();
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endlist' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginlist'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::begintuple(int64_t numfields) {
    if (length_ == -1) {
      for (int64_t i = 0;  i < numfields;  i++) {
        contents_.push_back(std::shared_ptr<Builder>(UnknownBuilder::fromempty(options_)));
      }
      length_ = 0;
    }

    if (!begun_  &&  numfields == (int64_t)contents_.size()) {
      begun_ = true;
      nextindex_ = -1;
    }
    else if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->begintuple(numfields);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'begintuple' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      nextindex_ = index;
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (contents_[i].get()->length() == length_) {
          maybeupdate(i, contents_[i].get()->null());
        }
        if (contents_[i].get()->length() != length_ + 1) {
          throw std::invalid_argument(std::string("tuple index ") + std::to_string(i) + std::string(" filled more than once"));
        }
      }
      length_++;
      begun_ = false;
    }
    else {
      contents_[(size_t)nextindex_].get()->endtuple();
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::beginrecord(const char* name, bool check) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->beginrecord(name, check);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginrecord' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginrecord(name, check));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(name, check);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'field_fast' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, check);
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endrecord' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return that_;
  }

  const std::shared_ptr<Builder> TupleBuilder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'append' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->append(array, at));
    }
    else {
      contents_[(size_t)nextindex_].get()->append(array, at);
    }
    return that_;
  }

  void TupleBuilder::maybeupdate(int64_t i, const std::shared_ptr<Builder>& tmp) {
    if (tmp.get() != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = tmp;
    }
  }
}
