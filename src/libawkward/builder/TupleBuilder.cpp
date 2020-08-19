// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TupleBuilder.cpp", line)

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
  const BuilderPtr
  TupleBuilder::fromempty(const ArrayBuilderOptions& options) {
    BuilderPtr out = std::make_shared<TupleBuilder>(options,
                                                    std::vector<BuilderPtr>(),
                                                    -1,
                                                    false,
                                                    -1);
    out.get()->setthat(out);
    return out;
  }

  TupleBuilder::TupleBuilder(const ArrayBuilderOptions& options,
                             const std::vector<BuilderPtr>& contents,
                             int64_t length,
                             bool begun,
                             size_t nextindex)
      : options_(options)
      , contents_(contents)
      , length_(length)
      , begun_(begun)
      , nextindex_((int64_t)nextindex) { }

  int64_t
  TupleBuilder::numfields() const {
    return (int64_t)contents_.size();
  }

  const std::string
  TupleBuilder::classname() const {
    return "TupleBuilder";
  };

  int64_t
  TupleBuilder::length() const {
    return length_;
  }

  void
  TupleBuilder::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    length_ = -1;
    begun_ = false;
    nextindex_ = -1;
  }

  const ContentPtr
  TupleBuilder::snapshot() const {
    if (length_ == -1) {
      return std::make_shared<EmptyArray>(Identities::none(),
                                          util::Parameters());
    }
    ContentPtrVec contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      contents.push_back(contents_[i].get()->snapshot());
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         util::Parameters(),
                                         contents,
                                         util::RecordLookupPtr(nullptr),
                                         length_);
  }

  bool
  TupleBuilder::active() const {
    return begun_;
  }

  const BuilderPtr
  TupleBuilder::null() {
    if (!begun_) {
      BuilderPtr out = OptionBuilder::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'null' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::boolean(bool x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->boolean(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'boolean' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::integer(int64_t x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->integer(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'integer' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::real(double x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->real(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'real' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->string(x, length, encoding);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'string' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->string(x,
                                                              length,
                                                              encoding));
    }
    else {
      contents_[(size_t)nextindex_].get()->string(x, length, encoding);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::beginlist() {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->beginlist();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'begin_list' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->beginlist());
    }
    else {
      contents_[(size_t)nextindex_].get()->beginlist();
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::endlist() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_list' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple' and then 'begin_list'")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::begintuple(int64_t numfields) {
    if (length_ == -1) {
      for (int64_t i = 0;  i < numfields;  i++) {
        contents_.push_back(BuilderPtr(UnknownBuilder::fromempty(options_)));
      }
      length_ = 0;
    }

    if (!begun_  &&  numfields == (int64_t)contents_.size()) {
      begun_ = true;
      nextindex_ = -1;
    }
    else if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->begintuple(numfields);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'begin_tuple' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'")
        + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1  ||
             !contents_[(size_t)nextindex_].get()->active()) {
      nextindex_ = index;
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::endtuple() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1  ||
             !contents_[(size_t)nextindex_].get()->active()) {
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (contents_[i].get()->length() == length_) {
          maybeupdate((int64_t)i, contents_[i].get()->null());
        }
        if (contents_[i].get()->length() != length_ + 1) {
          throw std::invalid_argument(
            std::string("tuple index ") + std::to_string(i)
            + std::string(" filled more than once") + FILENAME(__LINE__));
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

  const BuilderPtr
  TupleBuilder::beginrecord(const char* name, bool check) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->beginrecord(name, check);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'begin_record' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->beginrecord(name, check)
                  );
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(name, check);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'field_fast' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'field_fast' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple' and then 'begin_record'")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, check);
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_record' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple' and then 'begin_record'")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return that_;
  }

  const BuilderPtr
  TupleBuilder::append(const ContentPtr& array, int64_t at) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'append' immediately after 'begin_tuple'; "
                    "needs 'index' or 'end_tuple'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->append(array, at));
    }
    else {
      contents_[(size_t)nextindex_].get()->append(array, at);
    }
    return that_;
  }

  void
  TupleBuilder::maybeupdate(int64_t i, const BuilderPtr& tmp) {
    if (tmp.get() != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = tmp;
    }
  }
}
