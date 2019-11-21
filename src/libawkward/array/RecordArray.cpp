// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
// #include "awkward/type/RecordType.h"

#include "awkward/array/RecordArray.h"

namespace awkward {
    const std::string RecordArray::classname() const {
      return "RecordArray";
    }

    void RecordArray::setid() {
      throw std::runtime_error("RecordArray::setid");
    }

    void RecordArray::setid(const std::shared_ptr<Identity> id) {
      throw std::runtime_error("RecordArray::setid");
    }

    const std::string RecordArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
      std::stringstream out;
      out << indent << pre << "<" << classname() << ">\n";
      if (id_.get() != nullptr) {
        out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
      }
      for (size_t i = 0;  i < contents_.size();  i++) {
        out << indent << "    <field i=\"" << i << "\"";
        if (reverselookup_.get() != nullptr) {
          out << " key=\"" << reverselookup_.get()->at(i) << "\">";
          for (auto pair : *lookup_.get()) {
            if (pair.second == i  &&  pair.first != reverselookup_.get()->at(i)) {
              out << "<alias>" << pair.first << "</alias>";
            }
          }
        }
        else {
          out << ">";
        }
        out << "\n" << indent;
        out << contents_[i].get()->tostring_part(indent + std::string("        "), "", "\n");
        out << indent << "    </field>\n";
      }
      out << indent << "</" << classname() << ">" << post;
      return out.str();
    }

    void RecordArray::tojson_part(ToJson& builder) const {
      int64_t len = length();
      for (int64_t i = 0;  i < len;  i++) {
        builder.beginrec();





        builder.endrec();
      }
    }

    const std::shared_ptr<Type> RecordArray::type_part() const {
      throw std::runtime_error("RecordArray::type_part");
    }

    int64_t RecordArray::length() const {
      throw std::runtime_error("RecordArray::length");
    }

    const std::shared_ptr<Content> RecordArray::shallow_copy() const {
      throw std::runtime_error("RecordArray::shallow_copy");
    }

    void RecordArray::check_for_iteration() const {
      throw std::runtime_error("RecordArray::check_for_iteration");
    }

    const std::shared_ptr<Content> RecordArray::getitem_nothing() const {
      throw std::runtime_error("RecordArray::getitem_nothing");
    }

    const std::shared_ptr<Content> RecordArray::getitem_at(int64_t at) const {
      throw std::runtime_error("RecordArray::getitem_at");
    }

    const std::shared_ptr<Content> RecordArray::getitem_at_nowrap(int64_t at) const {
      throw std::runtime_error("RecordArray::getitem_at_nowrap");
    }

    const std::shared_ptr<Content> RecordArray::getitem_range(int64_t start, int64_t stop) const {
      throw std::runtime_error("RecordArray::getitem_range");
    }

    const std::shared_ptr<Content> RecordArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
      throw std::runtime_error("RecordArray::getitem_range_nowrap");
    }

    const std::shared_ptr<Content> RecordArray::carry(const Index64& carry) const {
      throw std::runtime_error("RecordArray::carry");
    }

    const std::pair<int64_t, int64_t> RecordArray::minmax_depth() const {
      throw std::runtime_error("RecordArray::minmax_depth");
    }

    int64_t RecordArray::numfields() const {
      return (int64_t)contents_.size();
    }

    const std::shared_ptr<Content> RecordArray::field(int64_t i) const {
      if (i >= numfields()) {
        throw std::invalid_argument(std::string("field ") + std::to_string(i) + std::string(" requested from RecordArray with only ") + std::to_string(numfields()) + std::string(" fields"));
      }
      return contents_[(size_t)i];
    }

    const std::shared_ptr<Content> RecordArray::field(const std::string& fieldname) const {
      if (lookup_.get() == nullptr) {
        throw std::invalid_argument("field requested by name from RecordArray without named fields");
      }
      size_t i;
      try {
        i = lookup_.get()->at(fieldname);
      }
      catch (std::out_of_range err) {
        throw std::invalid_argument(std::string("fieldname \"") + fieldname + std::string("\" is not in RecordArray"));
      }
      if (i >= contents_.size()) {
        throw std::invalid_argument(std::string("fieldname \"") + fieldname + std::string("\" points to tuple index ") + std::to_string(i) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
      }
      return contents_[i];
    }

    void RecordArray::append(const std::shared_ptr<Content>& content, const std::string& key) {
      size_t i = contents_.size();
      append(content);
      setkey(i, key);
    }

    void RecordArray::append(const std::shared_ptr<Content>& content) {
      if (reverselookup_.get() != nullptr) {
        reverselookup_.get()->push_back(std::to_string(contents_.size()));
      }
      contents_.push_back(content);
    }

    void RecordArray::setkey(int64_t i, const std::string& fieldname) {
      if (lookup_.get() == nullptr) {
        lookup_ = std::shared_ptr<Lookup>(new Lookup);
        reverselookup_ = std::shared_ptr<ReverseLookup>(new ReverseLookup);
        for (size_t i = 0;  i < contents_.size();  i++) {
          reverselookup_.get()->push_back(std::to_string(i));
        }
      }
      (*lookup_.get())[fieldname] = i;
      (*reverselookup_.get())[i] = fieldname;
    }

    const std::shared_ptr<Content> RecordArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("RecordArray::getitem_next");
    }

    const std::shared_ptr<Content> RecordArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("RecordArray::getitem_next");
    }

    const std::shared_ptr<Content> RecordArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
      throw std::runtime_error("RecordArray::getitem_next");
    }

}
