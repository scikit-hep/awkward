// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/RecordBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/UnknownBuilder.h"
#include "awkward/util.h"

#include "awkward/builder/RecordBuilder.h"

namespace awkward {
  const BuilderPtr
  RecordBuilder::fromempty(const BuilderOptions& options) {
    return std::make_shared<RecordBuilder>(options,
                                           std::vector<BuilderPtr>(),
                                           std::vector<std::string>(),
                                           std::vector<const char*>(),
                                           "",
                                           nullptr,
                                           -1,
                                           false,
                                           -1,
                                           -1);
  }

  RecordBuilder::RecordBuilder(const BuilderOptions& options,
                               const std::vector<BuilderPtr>& contents,
                               const std::vector<std::string>& keys,
                               const std::vector<const char*>& pointers,
                               const std::string& name,
                               const char* nameptr,
                               int64_t length,
                               bool begun,
                               int64_t nextindex,
                               int64_t nexttotry)
      : options_(options)
      , contents_(contents)
      , keys_(keys)
      , pointers_(pointers)
      , name_(name)
      , nameptr_(nameptr)
      , length_(length)
      , begun_(begun)
      , nextindex_(nextindex)
      , nexttotry_(nexttotry)
      , keys_size_((int64_t)keys.size()) { }

  const std::string
  RecordBuilder::name() const {
    return name_;
  }

  const char*
  RecordBuilder::nameptr() const {
    return nameptr_;
  }

  const std::string
  RecordBuilder::classname() const {
    return "RecordBuilder";
  };

  const std::string
  RecordBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    std::stringstream out;
    out << "{\"class\": \"RecordArray\", \"contents\": {";
    for (size_t i = 0;  i < contents_.size();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << "" + util::quote(keys_[i]) + ": ";
      out << contents_[i].get()->to_buffers(container, form_key_id);
    }
    out << "}, ";
    if (!name_.empty()) {
      out << "\"parameters\": {\"__record__\": " + util::quote(name_) + "}, ";
    }
    out << "\"form_key\": \"" + form_key.str() + "\"}";

    return out.str();
  }

  int64_t
  RecordBuilder::length() const {
    return length_;
  }

  void
  RecordBuilder::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    keys_.clear();
    pointers_.clear();
    name_ = "";
    nameptr_ = nullptr;
    length_ = -1;
    begun_ = false;
    nextindex_ = -1;
    nexttotry_ = 0;
    keys_size_ = 0;
  }

  bool
  RecordBuilder::active() const {
    return begun_;
  }

  const BuilderPtr
  RecordBuilder::null() {
    if (!begun_) {
      BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
      out.get()->null();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'null' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::boolean(bool x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->boolean(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'boolean' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::integer(int64_t x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->integer(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'integer' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::real(double x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->real(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'real' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::complex(std::complex<double> x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->complex(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'complex' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->complex(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->complex(x);
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::datetime(int64_t x, const std::string& unit) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->datetime(x, unit);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'datetime' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->datetime(x, unit));
    }
    else {
      contents_[(size_t)nextindex_].get()->datetime(x, unit);
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::timedelta(int64_t x, const std::string& unit) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->timedelta(x, unit);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'timedelta' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->timedelta(x, unit));
    }
    else {
      contents_[(size_t)nextindex_].get()->timedelta(x, unit);
    }
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->string(x, length, encoding);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'string' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
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
    return nullptr;
  }

  const BuilderPtr
  RecordBuilder::beginlist() {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->beginlist();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'begin_list' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record'") + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->beginlist());
    }
    else {
      contents_[(size_t)nextindex_].get()->beginlist();
    }
    return shared_from_this();
  }

  const BuilderPtr
  RecordBuilder::endlist() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_list' immediately after 'begin_record'; "
                    "needs 'index' or 'end_record' and then 'begin_list'")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return shared_from_this();
  }

  const BuilderPtr
  RecordBuilder::begintuple(int64_t numfields) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->begintuple(numfields);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'begin_tuple' immediately after 'begin_record'; "
                    "needs 'field_fast', 'field_check', or 'end_record'")
        + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return shared_from_this();
  }

  const BuilderPtr
  RecordBuilder::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'index' immediately after 'begin_record'; "
                    "needs 'field_fast', 'field_check' or 'end_record' "
                    "and then 'begin_tuple'") + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return shared_from_this();
  }

  const BuilderPtr
  RecordBuilder::endtuple() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' immediately after 'begin_record'; "
                    "needs 'field_fast', 'field_check', or 'end_record' "
                    "and then 'begin_tuple'") + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)nextindex_].get()->endtuple();
    }
    return shared_from_this();
  }

  const BuilderPtr
  RecordBuilder::beginrecord(const char* name, bool check) {
    if (length_ == -1) {
      if (name == nullptr) {
        name_ = std::string("");
      }
      else {
        name_ = std::string(name);
      }
      nameptr_ = check ? nullptr : name;
      length_ = 0;
    }

    if (!begun_  &&
        ((check  &&  name_ == name)  ||  (!check  &&  nameptr_ == name))) {
      begun_ = true;
      nextindex_ = -1;
      nexttotry_ = 0;
    } else if (!begun_  &&
        ((!check && nameptr_ == nullptr && name_ == name))) {
      begun_ = true;
      nextindex_ = -1;
      nexttotry_ = 0;

      // Rebuild pointer for this name
      nameptr_ = name;
    }
    else if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->beginrecord(name, check);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument(
        std::string("called 'begin_record' immediately after 'begin_record'; "
                    "needs 'field_fast', 'field_check', or 'end_record'")
        + FILENAME(__LINE__));
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_,
                  contents_[(size_t)nextindex_].get()->beginrecord(name,
                                                                   check));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(name, check);
    }
    return shared_from_this();
  }

  void
  RecordBuilder::field(const char* key, bool check) {
    if (check) {
      field_check(key);
    }
    else {
      field_fast(key);
    }
  }

  void
  RecordBuilder::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1  ||
             !contents_[(size_t)nextindex_].get()->active()) {
      int64_t i = nexttotry_;
      do {
        if (i >= keys_size_) {
          i = 0;
          if (i == nexttotry_) {
            break;
          }
        }
        if (pointers_[(size_t)i] == key) {
          nextindex_ = i;
          nexttotry_ = i + 1;
          return;
        }
        // If we have yet to see this field with `field_fast`, rebuild the pointer
        else if (pointers_[(size_t)i] == nullptr) {
           if (keys_[(size_t)i].compare(key) == 0) {
              nextindex_ = i;
              nexttotry_ = i + 1;
              pointers_[(size_t)i] = key;
              return;
            }
        }
        i++;
      } while (i != nexttotry_);
      nextindex_ = keys_size_;
      nexttotry_ = 0;
      if (length_ == 0) {
        contents_.push_back(UnknownBuilder::fromempty(options_));
      }
      else {
        contents_.push_back(
          OptionBuilder::fromnulls(options_,
                                   length_,
                                   UnknownBuilder::fromempty(options_)));
      }
      keys_.push_back(std::string(key));
      pointers_.push_back(key);
      keys_size_ = (int64_t)keys_.size();
      return;
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, false);
      return;
    }
  }

  void
  RecordBuilder::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1  ||
             !contents_[(size_t)nextindex_].get()->active()) {
      int64_t i = nexttotry_;
      do {
        if (i >= keys_size_) {
          i = 0;
          if (i == nexttotry_) {
            break;
          }
        }
        if (keys_[(size_t)i].compare(key) == 0) {
          nextindex_ = i;
          nexttotry_ = i + 1;
          return;
        }
        i++;
      } while (i != nexttotry_);
      nextindex_ = keys_size_;
      nexttotry_ = 0;
      if (length_ == 0) {
        contents_.emplace_back(UnknownBuilder::fromempty(options_));
      }
      else {
        contents_.emplace_back(
          OptionBuilder::fromnulls(options_,
                                   length_,
                                   UnknownBuilder::fromempty(options_)));
      }
      keys_.emplace_back(std::string(key));
      pointers_.emplace_back(nullptr);
      keys_size_ = (int64_t)keys_.size();
      return;
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, true);
      return;
    }
  }

  const BuilderPtr
  RecordBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it") + FILENAME(__LINE__));
    }
    else if (nextindex_ == -1  ||
             !contents_[(size_t)nextindex_].get()->active()) {
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (contents_[i].get()->length() == length_) {
          maybeupdate((int64_t)i, contents_[i].get()->null());
        }
        if (contents_[i].get()->length() != length_ + 1) {
          throw std::invalid_argument(
            std::string("record field ") + util::quote(keys_[i])
            + std::string(" filled more than once") + FILENAME(__LINE__));
        }
      }
      length_++;
      begun_ = false;
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return nullptr;
  }

  void
  RecordBuilder::maybeupdate(int64_t i, const BuilderPtr tmp) {
    if (tmp  &&  tmp.get() != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = std::move(tmp);
    }
  }
}
