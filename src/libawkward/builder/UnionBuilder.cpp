// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/UnionBuilder.cpp", line)

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/type/UnionType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/BoolBuilder.h"
#include "awkward/builder/Int64Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/StringBuilder.h"
#include "awkward/builder/ListBuilder.h"
#include "awkward/builder/TupleBuilder.h"
#include "awkward/builder/RecordBuilder.h"
#include "awkward/builder/IndexedBuilder.h"
#include "awkward/array/UnionArray.h"

#include "awkward/builder/UnionBuilder.h"

namespace awkward {
  const BuilderPtr
  UnionBuilder::fromsingle(const ArrayBuilderOptions& options,
                           const BuilderPtr& firstcontent) {
    GrowableBuffer<int8_t> tags =
      GrowableBuffer<int8_t>::full(options, 0, firstcontent->length());
    GrowableBuffer<int64_t> index =
      GrowableBuffer<int64_t>::arange(options, firstcontent->length());
    std::vector<BuilderPtr> contents({ firstcontent });
    BuilderPtr out = std::make_shared<UnionBuilder>(options,
                                                    tags,
                                                    index,
                                                    contents);
    out.get()->setthat(out);
    return out;
  }

  UnionBuilder::UnionBuilder(const ArrayBuilderOptions& options,
                             const GrowableBuffer<int8_t>& tags,
                             const GrowableBuffer<int64_t>& index,
                             std::vector<BuilderPtr>& contents)
      : options_(options)
      , tags_(tags)
      , index_(index)
      , contents_(contents)
      , current_(-1) { }

  const std::string
  UnionBuilder::classname() const {
    return "UnionBuilder";
  };

  int64_t
  UnionBuilder::length() const {
    return tags_.length();
  }

  void
  UnionBuilder::clear() {
    tags_.clear();
    index_.clear();
    for (auto x : contents_) {
      x.get()->clear();
    }
  }

  const ContentPtr
  UnionBuilder::snapshot() const {
    Index8 tags(tags_.ptr(), 0, tags_.length(), kernel::lib::cpu);
    Index64 index(index_.ptr(), 0, index_.length(), kernel::lib::cpu);
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->snapshot());
    }
    return UnionArray8_64(Identities::none(),
                          util::Parameters(),
                          tags,
                          index,
                          contents).simplify_uniontype(false);
  }

  bool
  UnionBuilder::active() const {
    return current_ != -1;
  }

  const BuilderPtr
  UnionBuilder::null() {
    if (current_ == -1) {
      BuilderPtr out = OptionBuilder::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else {
      contents_[(size_t)current_].get()->null();
      return that_;
    }
  }

  const BuilderPtr
  UnionBuilder::boolean(bool x) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<BoolBuilder*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = BoolBuilder::fromempty(options_);
        contents_.push_back(tofill);
      }
      int64_t length = tofill.get()->length();
      tofill.get()->boolean(x);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->boolean(x);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::integer(int64_t x) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<Int64Builder*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = Int64Builder::fromempty(options_);
        contents_.push_back(tofill);
      }
      int64_t length = tofill.get()->length();
      tofill.get()->integer(x);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->integer(x);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::real(double x) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<Float64Builder*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        i = 0;
        for (auto content : contents_) {
          if (dynamic_cast<Int64Builder*>(content.get()) != nullptr) {
            tofill = content;
            break;
          }
          i++;
        }
        if (tofill.get() != nullptr) {
          tofill = Float64Builder::fromint64(
            options_,
            dynamic_cast<Int64Builder*>(tofill.get())->buffer());
          contents_[(size_t)i] = tofill;
        }
        else {
          tofill = Float64Builder::fromempty(options_);
          contents_.push_back(tofill);
        }
      }
      int64_t length = tofill.get()->length();
      tofill.get()->real(x);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->real(x);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (StringBuilder* raw = dynamic_cast<StringBuilder*>(content.get())) {
          if (raw->encoding() == encoding) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = StringBuilder::fromempty(options_, encoding);
        contents_.push_back(tofill);
      }
      int64_t len = tofill.get()->length();
      tofill.get()->string(x, length, encoding);
      tags_.append(i);
      index_.append(len);
    }
    else {
      contents_[(size_t)current_].get()->string(x, length, encoding);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::beginlist() {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<ListBuilder*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = ListBuilder::fromempty(options_);
        contents_.push_back(tofill);
      }
      tofill->beginlist();
      current_ = i;
    }
    else {
      contents_[(size_t)current_].get()->beginlist();
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::endlist() {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endlist();
      if (length != contents_[(size_t)current_].get()->length()) {
        tags_.append(current_);
        index_.append(length);
        current_ = -1;
      }
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::begintuple(int64_t numfields) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (TupleBuilder* raw = dynamic_cast<TupleBuilder*>(content.get())) {
          if (raw->length() == -1  ||  raw->numfields() == numfields) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = TupleBuilder::fromempty(options_);
        contents_.push_back(tofill);
      }
      tofill->begintuple(numfields);
      current_ = i;
    }
    else {
      contents_[(size_t)current_].get()->begintuple(numfields);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::index(int64_t index) {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)current_].get()->index(index);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::endtuple() {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endtuple();
      if (length != contents_[(size_t)current_].get()->length()) {
        tags_.append(current_);
        index_.append(length);
        current_ = -1;
      }
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::beginrecord(const char* name, bool check) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (RecordBuilder* raw = dynamic_cast<RecordBuilder*>(content.get())) {
          if (raw->length() == -1  ||
              ((check  &&  raw->name() == name)  ||
               (!check  &&  raw->nameptr() == name))) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = RecordBuilder::fromempty(options_);
        contents_.push_back(tofill);
      }
      tofill->beginrecord(name, check);
      current_ = i;
    }
    else {
      contents_[(size_t)current_].get()->beginrecord(name, check);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::field(const char* key, bool check) {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)current_].get()->field(key, check);
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::endrecord() {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it") + FILENAME(__LINE__));
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endrecord();
      if (length != contents_[(size_t)current_].get()->length()) {
        tags_.append(current_);
        index_.append(length);
        current_ = -1;
      }
    }
    return that_;
  }

  const BuilderPtr
  UnionBuilder::append(const ContentPtr& array, int64_t at) {
    if (current_ == -1) {
      BuilderPtr tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (IndexedGenericBuilder* raw =
            dynamic_cast<IndexedGenericBuilder*>(content.get())) {
          if (raw->arrayptr() == array.get()) {
            tofill = content;
            break;
          }
        }
        else if (IndexedI32Builder* raw =
                 dynamic_cast<IndexedI32Builder*>(content.get())) {
          if (raw->arrayptr() == array.get()) {
            tofill = content;
            break;
          }
        }
        else if (IndexedIU32Builder* raw =
                 dynamic_cast<IndexedIU32Builder*>(content.get())) {
          if (raw->arrayptr() == array.get()) {
            tofill = content;
            break;
          }
        }
        else if (IndexedI64Builder* raw =
                 dynamic_cast<IndexedI64Builder*>(content.get())) {
          if (raw->arrayptr() == array.get()) {
            tofill = content;
            break;
          }
        }
        else if (IndexedIO32Builder* raw =
                 dynamic_cast<IndexedIO32Builder*>(content.get())) {
          if (raw->arrayptr() == array.get()) {
            tofill = content;
            break;
          }
        }
        else if (IndexedIO64Builder* raw =
                 dynamic_cast<IndexedIO64Builder*>(content.get())) {
          if (raw->arrayptr() == array.get()) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = IndexedGenericBuilder::fromnulls(options_, 0, array);
        contents_.push_back(tofill);
      }
      int64_t length = tofill.get()->length();
      tofill.get()->append(array, at);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->append(array, at);
    }
    return that_;
  }
}
