#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

//#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("../src/awkward/_v2/cpp-headers/LayoutBuilder.h", line)

#include "GrowableBuffer.h"
//#include "rdataframe_jagged_builders.h"

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include <complex>

namespace awkward {
  /* struct Record
  {

  }; */

  template <typename T>
  const std::string
  type_to_name() {
      return typeid(T).name();
  }

  template <>
  const std::string
  type_to_name<bool>() {
      return "bool";
  }

  template <>
  const std::string
  type_to_name<int8_t>() {
      return "int8";
  }

  template <>
  const std::string
  type_to_name<int16_t>() {
      return "int16";
  }

  template <>
  const std::string
  type_to_name<int32_t>() {
      return "int32";
  }

  template <>
  const std::string
  type_to_name<int64_t>() {
      return "int64";
  }

  template <>
  const std::string
  type_to_name<uint8_t>() {
      return "uint8";
  }

  template <>
  const std::string
  type_to_name<uint16_t>() {
      return "uint16";
  }

  template <>
  const std::string
  type_to_name<uint32_t>() {
      return "uint32";
  }

  template <>
  const std::string
  type_to_name<uint64_t>() {
      return "uint64";
  }

  template <>
  const std::string
  type_to_name<float>() {
      return "float32";
  }

  template <>
  const std::string
  type_to_name<double>() {
      return "float64";
  }

  template <>
  const std::string
  type_to_name<char>() {
      return "char";
  }

  template <typename Test, template <typename...> class Ref>
  struct is_specialization : std::false_type {
  };

  template <template <typename...> class Ref, typename... Args>
  struct is_specialization<Ref<Args...>, Ref> : std::true_type {
  };

  template <typename PRIMITIVE>
  class NumpyLayoutBuilder {
  public:
    NumpyLayoutBuilder(size_t initial = 1024)
        : data_(awkward::GrowableBuffer<PRIMITIVE>(initial)) { }

    void
    append(PRIMITIVE x) {
      data_.append(x);
    }

    void
    append(PRIMITIVE* ptr, size_t size) {
      for (int64_t i = 0; i < size; i++) {
        data_.append(ptr[i]);
      }
    }

    int64_t
    length() const {
      return data_.length();
    }

    void
    clear() {
      data_.clear();
    }

    void
    null() {
      data_.append(-1);
    }

    PRIMITIVE*
    to_buffers() const {
      PRIMITIVE* ptr = new PRIMITIVE[length()];
      data_.concatenate(ptr);
      return ptr;
    }

    std::string
    form(int64_t form_key_id = -1) {
      std::stringstream form_key;
      form_key << "node" << (++form_key_id);
      if (std::is_arithmetic<PRIMITIVE>::value) {
        return "{\"class\": \"NumpyArray\", \"primitive\": \""
          + type_to_name<PRIMITIVE>() + "\", " + "\"form_key\": \"" + form_key.str() + "\"}";
      } else if (is_specialization<PRIMITIVE, std::complex>::value) {
        return "{\"class\": \"NumpyArray\", \"primitive\": \"complex128\", \"form_key\": \""
          + form_key.str() + "\"}";
      }
      return "unsupported type";
    }

    void
    dump(std::string indent) const {
      std::cout << indent << "NumpyLayoutBuilder" << std::endl;
      std::cout << indent << "  data ";
      auto ptr = to_buffers();
      data_.dump(ptr);
      std::cout << std::endl;
    }

  private:
    size_t initial_;
    awkward::GrowableBuffer<PRIMITIVE> data_;
  };

  template <typename BUILDER>
  class ListOffsetLayoutBuilder {
  public:
    ListOffsetLayoutBuilder(size_t initial = 1024)
        : offsets_(awkward::GrowableBuffer<int64_t>(initial))
        , begun_(false) {
      offsets_.append(0);
    }

    // returns JSON string
    std::string
    form(int64_t form_key_id = -1) {
      std::stringstream form_key;
      form_key << "node" << (++form_key_id);
      return "{\"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
      + content_.form(form_key_id) + ", " + "\"form_key\": \"" + form_key.str() + "\"}";
    }

    BUILDER*
    begin_list() {
      if (!begun_) {
        begun_ = true;
      }
      return &content_;
    }

    void
    end_list() {
      if (!begun_) {
        throw std::invalid_argument(
          std::string("called 'end_list' without 'begin_list' at the same level before it"
          "in ListOffsetLayoutBuilder"));
      } else {
        offsets_.append(content_.length());
        begun_ = false;
      }
    }

    void
    clear() {
      offsets_.clear();
      offsets_.append(0);
      content_.clear();
      begun_ = false;
    }

    int64_t
    length() const {
      return content_.length();
    }

    void
    null() {
      content_.append(-1);
    }

    int64_t*
    to_buffers() const {
      int64_t* ptr = new int64_t[offsets_.length()];
      offsets_.concatenate(ptr);
      return ptr;
    }

    void
    dump(std::string indent) const {
      std::cout << indent << "ListOffsetLayoutBuilder" << std::endl;
      std::cout << indent << "    offsets ";
      auto ptr = to_buffers();
      offsets_.dump(ptr);
      std::cout << std::endl;
      content_.dump("    ");
    }

  private:
    size_t initial_;
    bool begun_;
    GrowableBuffer<int64_t> offsets_;
    BUILDER content_;
  };

  template <typename BUILDER>
  class RecordLayoutBuilder {
  public:
    RecordLayoutBuilder()
        : contents_(std::vector<BUILDER*>()),
          index_(std::vector<int64_t>()),
          length_(-1),
          begun_(false),
          nextindex_(-1),
          nexttotry_(-1),
          index_size_(0) { }

    int64_t
    length() const {
      return length_;
    }

    void
    clear() {
      for (auto contents : contents_) {
        contents->clear();
      }
      index_.clear();
      length_ = -1;
      begun_ = false;
      nextindex_ = -1;
      nexttotry_ = 0;
      index_size_ = 0;
    }

    std::string
    form(int64_t form_key_id = -1) {
      std::stringstream form_key;
      form_key << "node" << (++form_key_id);
      std::stringstream out;
      out << "{\"class\": \"RecordArray\", \"contents\": {";
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (i != 0) {
          out << ", ";
        }
        field(0);
        out << "\"" + std::to_string(index_[i]) + "\": ";
        out << contents_[(size_t)i]->form(form_key_id);
      }
      out << "}, ";
      out << "\"form_key\": \"" + form_key.str() + "\"}";
      return out.str();
    }

    void
    begin_record() {
      if (length_ == -1) {
        length_ = 0;
      }
      if (!begun_) {
        begun_ = true;
        nextindex_ = -1;
        nexttotry_ = 0;
      }
    }

    BUILDER*
    field(int64_t index) {
      int64_t i = nexttotry_;
      do {
        if (i >= index_size_) {
          i = 0;
          if (i == nexttotry_) {
            break;
          }
        }
        if (index_[i] == index) {
          nextindex_ = i;
          nexttotry_ = i + 1;
          return contents_[nextindex_];
        }
        i++;
      } while (i != nexttotry_);

      nextindex_ = index_size_;
      nexttotry_ = 0;
      contents_.push_back(new BUILDER);  //vector of unique_ptr of Record
      index_.push_back(index);
      index_size_ = (int64_t)index_.size();
      return contents_[nextindex_];
    }

    void
    end_record() {
      if (!begun_) {
        throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it"));
      } else if (nextindex_ == -1) {
        for (size_t i = 0; i < contents_.size(); i++) {
          if (contents_[(size_t)i]->length() == length_) {
            //maybeupdate((int64_t)i, contents_[(size_t)i]);
          }
          if (contents_[(size_t)i]->length() != length_ + 1) {
            throw std::invalid_argument(
                  std::string("record field '") + std::to_string(index_[i]) +
                  std::string("' filled more than once in RecordLayoutBuilder"));
          }
        }
        length_++;
        begun_ = false;
      }
    }

    void
    dump(std::string indent) const {
      std::cout << indent << "RecordLayoutBuilder" << std::endl;
      int64_t i = 0;
      for (auto contents : contents_) {
        std::cout << indent << "  field " << index_[i] << std::endl;
        contents->dump("  ");
        i++;
      }
    }

  private:
    std::vector<BUILDER*> contents_;
    std::vector<int64_t> index_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;
    int64_t nexttotry_;
    int64_t index_size_;
  };
}  // namespace awkward

#endif  // AWKWARD_LAYOUTBUILDER_H_
