// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

#include "GrowableBuffer.h"
#include "utils.h"

#include <stdexcept>
#include <cassert>
#include <tuple>

namespace awkward {

  template<const char *str>
  struct field_name {
    const char* value = str;
  };

  template<class field_name, class BUILDER>
  struct Record {
    const char* field() {
        return field_.value;
    }

    std::string
    form() {
      return builder.form();
    }

    field_name field_;
    BUILDER builder;
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
  void
  visit_at(std::tuple<RECORDs...> const& contents, size_t index, FUNCTION fun)
  {
      visit_impl<sizeof...(RECORDs)>::visit(contents, index, fun);
  }

  template <typename FUNCTION, typename... RECORDs>
  void
  visit_at(std::tuple<RECORDs...>& contents, size_t index, FUNCTION fun)
  {
      visit_impl<sizeof...(RECORDs)>::visit(contents, index, fun);
  }

  template <unsigned ID, unsigned INITIAL, typename PRIMITIVE>
  class NumpyLayoutBuilder {
  public:
    NumpyLayoutBuilder()
        : data_(awkward::GrowableBuffer<PRIMITIVE>(INITIAL)) { }

    size_t
    length() const {
      return data_.length();
    }

    void
    clear() {
      data_.clear();
    }

    void
    append(PRIMITIVE x) {
      data_.append(x);
    }

    void
    append(PRIMITIVE* ptr, size_t size) {
      data_.append(ptr, size);
    }

    void
    to_buffers(PRIMITIVE* ptr) const {
      data_.concatenate(ptr);
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      if (std::is_arithmetic<PRIMITIVE>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \""
                  + type_to_name<PRIMITIVE>() + "\", " + "\"form_key\": \"" + form_key.str() + "\" }";
      } else if (is_specialization<PRIMITIVE, std::complex>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \"complex128\", \"form_key\": \""
                  + form_key.str() + "\" }";
      }
      return "unsupported type";
    }

  private:
    size_t initial_;
    awkward::GrowableBuffer<PRIMITIVE> data_;
    unsigned id_ = ID;
  };

  template <unsigned ID, unsigned INITIAL, typename BUILDER>
  class ListOffsetLayoutBuilder {
  public:
    ListOffsetLayoutBuilder()
        : offsets_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , begun_(false) {
      offsets_.append(0);
    }

    size_t
    length() const {
      return offsets_.length() - 1;
    }

    void
    clear() {
      offsets_.clear();
      offsets_.append(0);
      content_.clear();
      begun_ = false;
    }

    BUILDER*
    content() {
      return &content_;
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE x) {
      content_.append(x);
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE* ptr, size_t size) {
      content_.append(ptr, size);
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
        begun_ = false;
        offsets_.append(content_.length());
      }
    }

    void
    to_buffers(int64_t* ptr) const {
      offsets_.concatenate(ptr);
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
                + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    bool begun_;
    GrowableBuffer<int64_t> offsets_;
    BUILDER content_;
    unsigned id_ = ID;
  };

  template <unsigned ID, typename... RECORD>
  class RecordLayoutBuilder {
  public:
    RecordLayoutBuilder()
        : contents({new RECORD}...)
        , length_(0)
        , begun_(false) { }

    size_t
    length() const {
      return length_;
    }

    void
    clear() {
      length_ = 0;
      begun_ = false;
      auto clear_contents = [](auto record) { record->builder.clear(); };
      for (size_t i = 0; i < std::tuple_size<decltype(contents)>::value; i++)
        visit_at(contents, i, clear_contents);
    }

    void
    begin_record() {
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

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
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

    std::tuple<RECORD*...> contents;

    private:
    size_t length_;
    bool begun_;
    unsigned id_ = ID;
  };


template <unsigned ID, unsigned INITIAL, typename BUILDER>
  class ListLayoutBuilder {
  public:
    ListLayoutBuilder()
        : starts_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , stops_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , begun_(false)
        , length_(0) { }

    size_t
    length() const {
      return length_;
    }

    void
    clear() {
      starts_.clear();
      stops_.clear();
      content_.clear();
      begun_ = false;
      length_ = 0;
    }

    BUILDER*
    content() {
      return &content_;
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE x) {
      content_.append(x);
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE* ptr, size_t size) {
      content_.append(ptr, size);
    }

    BUILDER*
    begin_list() {
      if (!begun_) {
        begun_ = true;
        starts_.append(content_.length());
      }
      return &content_;
    }

    void
    end_list() {
      if (!begun_) {
        throw std::invalid_argument(
          std::string("called 'end_list' without 'begin_list' at the same level before it"
          "in ListLayoutBuilder"));
      } else {
        length_++;
        begun_ = false;
        stops_.append(content_.length());
      }
    }

    void
    to_buffers(int64_t* starts, int64_t* stops) const {
      starts_.concatenate(starts);
      stops_.concatenate(stops);
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"ListArray\", \"starts\": \"i64\", \"stops\": \"i64\", \"content\": "
                + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    bool begun_;
    size_t length_;
    GrowableBuffer<int64_t> starts_;
    GrowableBuffer<int64_t> stops_;
    BUILDER content_;
    unsigned id_ = ID;
  };

template <unsigned ID, unsigned INITIAL, typename BUILDER>
  class IndexedLayoutBuilder {
  public:
    IndexedLayoutBuilder()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL)) { }

    size_t
    length() const {
      return index_.length();
    }

    void
    clear() {
      index_.clear();
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE x) {
      index_.append(content_.length());
      content_.append(x);
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE* ptr, size_t size) {
      for (size_t i = 0; i < size; i++) {
        index_.append(content_.length());
        content_.append(ptr[i]);
      }
    }

    void
    to_buffers(int64_t* ptr) const {
      index_.concatenate(ptr);
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"IndexedArray\", \"index\": \"i64\", \"content\": "
                + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> index_;
    BUILDER content_;
    unsigned id_ = ID;
  };

template <unsigned ID, unsigned INITIAL, typename BUILDER>
  class IndexedOptionLayoutBuilder {
  public:
    IndexedOptionLayoutBuilder()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL)) { }

    size_t
    length() const {
      return index_.length();
    }

    void
    clear() {
      index_.clear();
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE x) {
      index_.append(content_.length());
      content_.append(x);
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE* ptr, size_t size) {
      for (size_t i = 0; i < size; i++) {
        index_.append(content_.length());
        content_.append(ptr[i]);
      }
    }

    void
    null() {
      index_.append(-1);
    }

    void
    to_buffers(int64_t* ptr) const {
      index_.concatenate(ptr);
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"IndexedOptionArray\", \"index\": \"i64\", \"content\": "
                + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> index_;
    BUILDER content_;
    unsigned id_ = ID;
  };

  template <unsigned ID>
  class EmptyLayoutBuilder {
  public:
    size_t
    length() const {
      return 0;
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"EmptyArray\" }";
    }

  private:
    unsigned id_ = ID;
  };

  template <unsigned ID, typename BUILDER>
  class UnmaskedLayoutBuilder {
  public:
    size_t
    length() const {
      return content_.length();
    }

    void
    clear() {
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE x) {
      content_.append(x);
    }

    template<typename PRIMITIVE>
    void
    append(PRIMITIVE* ptr, size_t size) {
      content_.append(ptr, size);
    }

    std::string
    form() {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"UnmaskedArray\", \"content\": " + content_.form()
                + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    BUILDER content_;
    unsigned id_ = ID;
  };
}  // namespace awkward

#endif  // AWKWARD_LAYOUTBUILDER_H_
