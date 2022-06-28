#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

//#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("../src/awkward/_v2/cpp-headers/LayoutBuilder.h", line)

#include "GrowableBuffer.h"
//#include "rdataframe_jagged_builders.h"

#include <stdexcept>
#include <cassert>
#include <stdint.h>
#include <string>
#include <vector>
#include <tuple>
#include <complex>

namespace awkward {
  extern int64_t form_key_id;
  int64_t form_key_id = -1;

  template<const char *str>
  struct field_name {
    const char* value = str;
  };

  template<class field_name, typename BUILDER>
  struct Record {
    const char* field() {
        return field_.value;
    }

    std::string
    form() {
      return(builder_.form());
    }

    field_name field_;
    BUILDER builder_;
  };

  template <size_t INDEX>
  struct visit_impl
  {
      template <typename RECORD, typename FUNCTION>
      static void visit(RECORD& contents, size_t index, FUNCTION fun)
      {
          if (index == INDEX - 1) fun(std::get<INDEX - 1>(contents));
          else visit_impl<INDEX - 1>::visit(contents, index, fun);
      }
  };

  template <>
  struct visit_impl<0>
  {
      template <typename RECORD, typename FUNCTION>
      static void visit(RECORD& contents, size_t index, FUNCTION fun) { assert(false); }
  };

  template <typename FUNCTION, typename... RECORDs>
  void visit_at(std::tuple<RECORDs...> const& contents, size_t index, FUNCTION fun)
  {
      visit_impl<sizeof...(RECORDs)>::visit(contents, index, fun);
  }

  template <typename FUNCTION, typename... RECORDs>
  void visit_at(std::tuple<RECORDs...>& contents, size_t index, FUNCTION fun)
  {
      visit_impl<sizeof...(RECORDs)>::visit(contents, index, fun);
  }

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

  template <unsigned INITIAL, typename PRIMITIVE>
  class NumpyLayoutBuilder {
  public:
    NumpyLayoutBuilder()
        : data_(awkward::GrowableBuffer<PRIMITIVE>(INITIAL)) { }

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

    PRIMITIVE*
    to_buffers() const {
      PRIMITIVE* ptr = new PRIMITIVE[length()];
      data_.concatenate(ptr);
      return ptr;
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << (++form_key_id);
      if (std::is_arithmetic<PRIMITIVE>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \""
          + type_to_name<PRIMITIVE>() + "\", " + "\"form_key\": \"" + form_key.str() + "\" }";
      } else if (is_specialization<PRIMITIVE, std::complex>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \"complex128\", \"form_key\": \""
          + form_key.str() + "\" }";
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

  template <unsigned INITIAL, typename BUILDER>
  class ListOffsetLayoutBuilder {
  public:
    ListOffsetLayoutBuilder()
        : offsets_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , begun_(false)
        , length_(0) {
      offsets_.append(0);
    }

    // returns JSON string
    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << (++form_key_id);
      return "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
      + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

    template<typename PRIMITIVE>
    void append(PRIMITIVE x) {
      content_.append(x);
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
      }
      else {
        length_++;
        begun_ = false;
        offsets_.append(content_.length());
      }
    }

    void
    clear() {
      offsets_.clear();
      offsets_.append(0);
      content_.clear();
      begun_ = false;
      length_ = 0;
    }

    int64_t
    length() const {
      return length_;
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
      content_.dump(indent + "    ");
    }

  private:
    bool begun_;
    int64_t length_;
    GrowableBuffer<int64_t> offsets_;
    BUILDER content_;
  };

  template <typename... RECORD>
  class RecordLayoutBuilder {
  public:
    RecordLayoutBuilder()
        : contents({new RECORD}...)
        , length_(-1)
        , begun_(false) { }

    int64_t
    length() const {
      return length_;
    }

    void
    clear() {
      auto clear_contents = [](auto record) { record->builder()->clear(); };
      for (size_t i = 0; i < 3; i++)
        visit_at( contents, i, clear_contents);
      length_ = -1;
      begun_ = false;
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << (++form_key_id);
      std::stringstream out;
      out << "{ \"class\": \"RecordArray\", \"contents\": { ";
      for (size_t i = 0;  i < std::tuple_size<decltype(contents)>::value;  i++) {
        if (i != 0) {
          out << ", ";
        }
        auto contents_form = [&out] (auto record) { out << "\"" << record->field() << + "\": ";
                                                 out << record->form(); };
        visit_at(contents, i, contents_form);
      }
      out << " }, ";
      out << "\"form_key\": \"" + form_key.str() + "\" }";
      return out.str();
    }

    void
    begin_record() {
      if (length_ == -1) {
        length_ = 0;
      }
      if (!begun_) {
        begun_ = true;
      }
    }

    void
    end_record() {
      if (!begun_) {
        throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it"));
      } else {
        length_++;
        begun_ = false;
      }
    }

    void
    dump(std::string indent) const {
      std::cout << indent << "RecordLayoutBuilder" << std::endl;
      auto print_contents = [&](auto record) { std::cout << indent << "  field " << record->field() << std::endl;
                                               record->builder()->dump(indent + "    "); };
      for (size_t i = 0; i < 3; i++) {
        visit_at( contents, i, print_contents);
      }
    }

    std::tuple<RECORD*...>  contents;

    private:
    int64_t length_;
    bool begun_;
  };

}  // namespace awkward

#endif  // AWKWARD_LAYOUTBUILDER_H_
