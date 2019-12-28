// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/type/UnionType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"
#include "awkward/fillable/Int64Fillable.h"
#include "awkward/fillable/Float64Fillable.h"
#include "awkward/fillable/StringFillable.h"
#include "awkward/fillable/ListFillable.h"
#include "awkward/fillable/TupleFillable.h"
#include "awkward/fillable/RecordFillable.h"

#include "awkward/fillable/UnionFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> UnionFillable::fromsingle(const FillableOptions& options, const std::shared_ptr<Fillable>& firstcontent) {
    GrowableBuffer<int8_t> types = GrowableBuffer<int8_t>::full(options, 0, firstcontent->length());
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::arange(options, firstcontent->length());
    std::vector<std::shared_ptr<Fillable>> contents({ firstcontent });
    std::shared_ptr<Fillable> out = std::make_shared<UnionFillable>(options, types, offsets, contents);
    out.get()->setthat(out);
    return out;
  }

  UnionFillable::UnionFillable(const FillableOptions& options, const GrowableBuffer<int8_t>& types, const GrowableBuffer<int64_t>& offsets, std::vector<std::shared_ptr<Fillable>>& contents)
      : options_(options)
      , types_(types)
      , offsets_(offsets)
      , contents_(contents)
      , current_(-1) { }

  const std::string UnionFillable::classname() const {
    return "UnionFillable";
  };

  int64_t UnionFillable::length() const {
    return types_.length();
  }

  void UnionFillable::clear() {
    types_.clear();
    offsets_.clear();
    for (auto x : contents_) {
      x.get()->clear();
    }
  }

  const std::shared_ptr<Type> UnionFillable::type() const {
    std::vector<std::shared_ptr<Type>> types;
    for (auto x : contents_) {
      types.push_back(x.get()->type());
    }
    return std::make_shared<UnionType>(util::Parameters(), types);
  }

  const std::shared_ptr<Content> UnionFillable::snapshot(const std::shared_ptr<Type>& type) const {
    Index8 types(types_.ptr(), 0, types_.length());
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    throw std::runtime_error("UnionFillable::snapshot needs UnionArray");
  }

  bool UnionFillable::active() const {
    return current_ != -1;
  }

  const std::shared_ptr<Fillable> UnionFillable::null() {
    if (current_ == -1) {
      std::shared_ptr<Fillable> out = OptionFillable::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else {
      contents_[(size_t)current_].get()->null();
      return that_;
    }
  }

  const std::shared_ptr<Fillable> UnionFillable::boolean(bool x) {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<BoolFillable*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = BoolFillable::fromempty(options_);
        contents_.push_back(tofill);
      }
      int64_t length = tofill.get()->length();
      tofill.get()->boolean(x);
      types_.append(i);
      offsets_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->boolean(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::integer(int64_t x) {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<Int64Fillable*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = Int64Fillable::fromempty(options_);
        contents_.push_back(tofill);
      }
      int64_t length = tofill.get()->length();
      tofill.get()->integer(x);
      types_.append(i);
      offsets_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->integer(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::real(double x) {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<Float64Fillable*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        i = 0;
        for (auto content : contents_) {
          if (dynamic_cast<Int64Fillable*>(content.get()) != nullptr) {
            tofill = content;
            break;
          }
          i++;
        }
        if (tofill.get() != nullptr) {
          tofill = Float64Fillable::fromint64(options_, dynamic_cast<Int64Fillable*>(tofill.get())->buffer());
          contents_[(size_t)i] = tofill;
        }
        else {
          tofill = Float64Fillable::fromempty(options_);
          contents_.push_back(tofill);
        }
      }
      int64_t length = tofill.get()->length();
      tofill.get()->real(x);
      types_.append(i);
      offsets_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->real(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::string(const char* x, int64_t length, const char* encoding) {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (StringFillable* raw = dynamic_cast<StringFillable*>(content.get())) {
          if (raw->encoding() == encoding) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = StringFillable::fromempty(options_, encoding);
        contents_.push_back(tofill);
      }
      int64_t len = tofill.get()->length();
      tofill.get()->string(x, length, encoding);
      types_.append(i);
      offsets_.append(len);
    }
    else {
      contents_[(size_t)current_].get()->string(x, length, encoding);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::beginlist() {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<ListFillable*>(content.get()) != nullptr) {
          tofill = content;
          break;
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = ListFillable::fromempty(options_);
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

  const std::shared_ptr<Fillable> UnionFillable::endlist() {
    if (current_ == -1) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endlist();
      if (length != contents_[(size_t)current_].get()->length()) {
        types_.append(current_);
        offsets_.append(length);
        current_ = -1;
      }
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::begintuple(int64_t numfields) {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (TupleFillable* raw = dynamic_cast<TupleFillable*>(content.get())) {
          if (raw->length() == -1  ||  raw->numfields() == numfields) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = TupleFillable::fromempty(options_);
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

  const std::shared_ptr<Fillable> UnionFillable::index(int64_t index) {
    if (current_ == -1) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      contents_[(size_t)current_].get()->index(index);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::endtuple() {
    if (current_ == -1) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endtuple();
      if (length != contents_[(size_t)current_].get()->length()) {
        types_.append(current_);
        offsets_.append(length);
        current_ = -1;
      }
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::beginrecord(const char* name, bool check) {
    if (current_ == -1) {
      std::shared_ptr<Fillable> tofill(nullptr);
      int8_t i = 0;
      for (auto content : contents_) {
        if (RecordFillable* raw = dynamic_cast<RecordFillable*>(content.get())) {
          if (raw->length() == -1  ||  ((check  &&  raw->name() == name)  ||  (!check  &&  raw->nameptr() == name))) {
            tofill = content;
            break;
          }
        }
        i++;
      }
      if (tofill.get() == nullptr) {
        tofill = RecordFillable::fromempty(options_);
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

  const std::shared_ptr<Fillable> UnionFillable::field(const char* key, bool check) {
    if (current_ == -1) {
      throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
    }
    else {
      contents_[(size_t)current_].get()->field(key, check);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> UnionFillable::endrecord() {
    if (current_ == -1) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endrecord();
      if (length != contents_[(size_t)current_].get()->length()) {
        types_.append(current_);
        offsets_.append(length);
        current_ = -1;
      }
    }
    return that_;
  }
}
