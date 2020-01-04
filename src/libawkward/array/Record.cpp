// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/ArrayType.h"

#include "awkward/array/Record.h"

namespace awkward {
  Record::Record(const RecordArray& array, int64_t at)
      : Content(Identities::none(), array.parameters())
      , array_(array)
      , at_(at) { }

  const std::shared_ptr<Content> Record::array() const {
    return array_.shallow_copy();
  }

  int64_t Record::at() const {
    return at_;
  }

  const std::vector<std::shared_ptr<Content>> Record::contents() const {
    std::vector<std::shared_ptr<Content>> out;
    for (auto item : array_.contents()) {
      out.push_back(item.get()->getitem_at_nowrap(at_));
    }
    return out;
  }

  const std::shared_ptr<util::RecordLookup> Record::recordlookup() const {
    return array_.recordlookup();
  }

  bool Record::istuple() const {
    return array_.istuple();
  }

  bool Record::isscalar() const {
    return true;
  }

  const std::string Record::classname() const {
    return "Record";
  }

  const std::shared_ptr<Identities> Record::identities() const {
    std::shared_ptr<Identities> recidentities = array_.identities();
    if (recidentities.get() == nullptr) {
      return recidentities;
    }
    else {
      return recidentities.get()->getitem_range_nowrap(at_, at_ + 1);
    }
  }

  void Record::setidentities() {
    throw std::runtime_error("undefined operation: Record::setidentities");
  }

  void Record::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("undefined operation: Record::setidentities");
  }

  const std::shared_ptr<Type> Record::type() const {
    std::shared_ptr<Type> out = array_.type();
    out.get()->setparameters(parameters_);
    return out;
  }

  const std::shared_ptr<Content> Record::astype(const std::shared_ptr<Type>& type) const {
    std::shared_ptr<Content> record = array_.astype(type);
    if (RecordArray* raw = dynamic_cast<RecordArray*>(record.get())) {
      if (raw->numfields() == 0) {
        return std::make_shared<Record>(RecordArray(raw->identities(), raw->parameters(), raw->length(), raw->istuple()), at_);
      }
      else {
        return std::make_shared<Record>(RecordArray(raw->identities(), raw->parameters(), raw->contents(), raw->recordlookup()), at_);
      }
    }
    else {
      throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring());
    }
  }

  const std::string Record::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " at=\"" << at_ << "\">\n";
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << array_.tostring_part(indent + std::string("    "), "", "\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void Record::tojson_part(ToJson& builder) const {
    size_t cols = (size_t)numfields();
    std::shared_ptr<util::RecordLookup> keys = array_.recordlookup();
    if (istuple()) {
      keys = std::make_shared<util::RecordLookup>();
      for (size_t j = 0;  j < cols;  j++) {
        keys.get()->push_back(std::to_string(j));
      }
    }
    std::vector<std::shared_ptr<Content>> contents = array_.contents();
    builder.beginrecord();
    for (size_t j = 0;  j < cols;  j++) {
      builder.field(keys.get()->at(j).c_str());
      contents[j].get()->getitem_at_nowrap(at_).get()->tojson_part(builder);
    }
    builder.endrecord();
  }

  int64_t Record::length() const {
    return -1;   // just like NumpyArray with ndim == 0, which is also a scalar
  }

  const std::shared_ptr<Content> Record::shallow_copy() const {
    return std::make_shared<Record>(array_, at_);
  }

  void Record::check_for_iteration() const {
    if (array_.identities().get() != nullptr  &&  array_.identities().get()->length() != 1) {
      util::handle_error(failure("len(identities) != 1 for scalar Record", kSliceNone, kSliceNone), array_.identities().get()->classname(), nullptr);
    }
  }

  const std::shared_ptr<Content> Record::getitem_nothing() const {
    throw std::runtime_error("undefined operation: Record::getitem_nothing");
  }

  const std::shared_ptr<Content> Record::getitem_at(int64_t at) const {
    throw std::invalid_argument(std::string("scalar Record can only be sliced by field name (string); try ") + util::quote(std::to_string(at), true));
  }

  const std::shared_ptr<Content> Record::getitem_at_nowrap(int64_t at) const {
    throw std::invalid_argument(std::string("scalar Record can only be sliced by field name (string); try ") + util::quote(std::to_string(at), true));
  }

  const std::shared_ptr<Content> Record::getitem_range(int64_t start, int64_t stop) const {
    throw std::invalid_argument("scalar Record can only be sliced by field name (string)");
  }

  const std::shared_ptr<Content> Record::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::invalid_argument("scalar Record can only be sliced by field name (string)");
  }

  const std::shared_ptr<Content> Record::getitem_field(const std::string& key) const {
    return array_.field(key).get()->getitem_at_nowrap(at_);
  }

  const std::shared_ptr<Content> Record::getitem_fields(const std::vector<std::string>& keys) const {
    RecordArray out(array_.identities(), parameters_, length(), istuple());
    if (istuple()) {
      for (auto key : keys) {
        out.append(array_.field(key));
      }
    }
    else {
      for (auto key : keys) {
        out.append(array_.field(key), key);
      }
    }
    return out.getitem_at_nowrap(at_);
  }

  const std::shared_ptr<Content> Record::carry(const Index64& carry) const {
    throw std::runtime_error("undefined operation: Record::carry");
  }

  const std::pair<int64_t, int64_t> Record::minmax_depth() const {
    return array_.minmax_depth();
  }

  int64_t Record::numfields() const {
    return array_.numfields();
  }

  int64_t Record::fieldindex(const std::string& key) const {
    return array_.fieldindex(key);
  }

  const std::string Record::key(int64_t fieldindex) const {
    return array_.key(fieldindex);
  }

  bool Record::haskey(const std::string& key) const {
    return array_.haskey(key);
  }

  const std::vector<std::string> Record::keys() const {
    return array_.keys();
  }

  const std::shared_ptr<Content> Record::field(int64_t fieldindex) const {
    return array_.field(fieldindex).get()->getitem_at_nowrap(at_);
  }

  const std::shared_ptr<Content> Record::field(const std::string& key) const {
    return array_.field(key).get()->getitem_at_nowrap(at_);
  }

  const std::vector<std::shared_ptr<Content>> Record::fields() const {
    std::vector<std::shared_ptr<Content>> out;
    int64_t cols = numfields();
    for (int64_t j = 0;  j < cols;  j++) {
      out.push_back(array_.field(j).get()->getitem_at_nowrap(at_));
    }
    return out;
  }

  const std::vector<std::pair<std::string, std::shared_ptr<Content>>> Record::fielditems() const {
    std::vector<std::pair<std::string, std::shared_ptr<Content>>> out;
    std::shared_ptr<util::RecordLookup> keys = array_.recordlookup();
    if (istuple()) {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(std::to_string(j), array_.field(j).get()->getitem_at_nowrap(at_)));
      }
    }
    else {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(keys.get()->at((size_t)j), array_.field(j).get()->getitem_at_nowrap(at_)));
      }
    }
    return out;
  }

  const Record Record::astuple() const {
    return Record(array_.astuple(), at_);
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: Record::getitem_next(at)");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: Record::getitem_next(range)");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: Record::getitem_next(array)");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: Record::getitem_next(field)");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: Record::getitem_next(fields)");
  }

}
