// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

#include "GrowableBuffer.h"
#include "utils.h"

#include <stdexcept>
#include <tuple>

namespace awkward {

  template<const char *str>
  struct field_name {
    const char* value = str;
  };

  template<class field_name, class BUILDER>
  struct Field {
    const char* field() {
        return field_.value;
    }

    std::string
    form() const noexcept {
      return builder.form();
    }

    field_name field_;
    BUILDER builder;
  };

  template <unsigned ID, unsigned INITIAL, typename PRIMITIVE>
  class NumpyLayoutBuilder {
  public:
    NumpyLayoutBuilder()
        : data_(awkward::GrowableBuffer<PRIMITIVE>(INITIAL)) { }

    size_t
    length() const noexcept {
      return data_.length();
    }

    void
    clear() noexcept {
      data_.clear();
    }

    void
    append(PRIMITIVE x) noexcept {
      data_.append(x);
    }

    void
    extend(PRIMITIVE* ptr, size_t size) noexcept {
      data_.extend(ptr, size);
    }

    void
    to_buffers(PRIMITIVE* ptr) const noexcept {
      data_.concatenate(ptr);
    }

    std::string
    form() const {
      std::stringstream form_key;
      form_key << "node" << id_;
      if (std::is_arithmetic<PRIMITIVE>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \""
                  + type_to_name<PRIMITIVE>()
                  + "\", \"form_key\": \"" + form_key.str() + "\" }";
      }
      else if (is_specialization<PRIMITIVE, std::complex>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \""
                  + type_to_name<PRIMITIVE>()
                  + "\", \"form_key\": \"" + form_key.str() + "\" }";
      }
      else {
        throw std::runtime_error("type " + std::string(typeid(PRIMITIVE).name()) + "is not supported");
      }
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
        : offsets_(awkward::GrowableBuffer<int64_t>(INITIAL)) {
      offsets_.append(0);
    }

    size_t
    length() const noexcept {
      return offsets_.length() - 1;
    }

    void
    clear() noexcept {
      offsets_.clear();
      offsets_.append(0);
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    BUILDER*
    begin_list() noexcept {
      return &content_;
    }

    void
    end_list() noexcept {
      offsets_.append(content_.length());
    }

    void
    to_buffers(int64_t* ptr) const noexcept {
      offsets_.concatenate(ptr);
    }

    std::string
    form() const noexcept {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
                + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> offsets_;
    BUILDER content_;
    unsigned id_ = ID;
  };

  template <unsigned ID, typename... RECORD>
  class RecordLayoutBuilder {
  public:
    RecordLayoutBuilder()
        : contents({new RECORD}...)
        , length_(0) { }

    size_t
    length() const noexcept {
      return length_;
    }

    void
    clear() noexcept {
      length_ = 0;
      auto clear_contents = [](auto record) { record->builder.clear(); };
      for (size_t i = 0; i < std::tuple_size<decltype(contents)>::value; i++)
        visit_at(contents, i, clear_contents);
    }

    void
    begin_record() {

    }

    void
    end_record() {
      length_++;
    }

    std::string
    form() const noexcept {
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
    unsigned id_ = ID;
  };


template <unsigned ID, unsigned INITIAL, typename BUILDER>
  class ListLayoutBuilder {
  public:
    ListLayoutBuilder()
        : starts_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , stops_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , length_(0) { }

    size_t
    length() const noexcept {
      return length_;
    }

    void
    clear() noexcept {
      starts_.clear();
      stops_.clear();
      content_.clear();
      length_ = 0;
    }

    BUILDER*
    content() {
      return &content_;
    }

    BUILDER*
    begin_list() noexcept {
      starts_.append(content_.length());
      return &content_;
    }

    void
    end_list() noexcept {
      length_++;
      stops_.append(content_.length());
    }

    void
    to_buffers(int64_t* starts, int64_t* stops) const noexcept {
      starts_.concatenate(starts);
      stops_.concatenate(stops);
    }

    std::string
    form() const noexcept {
      std::stringstream form_key;
      form_key << "node" << id_;
      return "{ \"class\": \"ListArray\", \"starts\": \"i64\", \"stops\": \"i64\", \"content\": "
                + content_.form() + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> starts_;
    GrowableBuffer<int64_t> stops_;
    BUILDER content_;
    size_t length_;
    unsigned id_ = ID;
  };

template <unsigned ID, unsigned INITIAL, typename BUILDER>
  class IndexedLayoutBuilder {
  public:
    IndexedLayoutBuilder()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL)) { }

    size_t
    length() const noexcept {
      return index_.length();
    }

    void
    clear() noexcept {
      index_.clear();
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    void
    append_index() noexcept {
      index_.append(content_.length());
    }

    void
    extend_index(size_t size) noexcept {
      size_t start = content_.length();
      for (size_t i = start; i < start + size; i++) {
        index_.append(i);
      }
    }

    void
    to_buffers(int64_t* ptr) const noexcept {
      index_.concatenate(ptr);
    }

    std::string
    form() const noexcept {
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
    length() const noexcept {
      return index_.length();
    }

    void
    clear() noexcept {
      index_.clear();
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    void
    append_index() noexcept {
      index_.append(content_.length());
    }

    void
    extend_index(size_t size) noexcept {
      size_t start = content_.length();
      for (size_t i = start; i < start + size; i++) {
        index_.append(i);
      }
    }

    void
    append_null() noexcept {
      index_.append(-1);
    }

    void
    extend_null(size_t size) noexcept {
      for (size_t i = 0; i < size; i++) {
        index_.append(-1);
      }
    }

    void
    null() {
      index_.append(-1);
    }

    void
    to_buffers(int64_t* ptr) const noexcept {
      index_.concatenate(ptr);
    }

    std::string
    form() const noexcept {
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
    length() const noexcept {
      return 0;
    }

    std::string
    form() const noexcept {
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
    length() const noexcept {
      return content_.length();
    }

    void
    clear() noexcept {
      content_.clear();
    }

    BUILDER*
    content() {
      return &content_;
    }

    template<typename PRIMITIVE>
    void
    append_valid() noexcept {
      return content_;
    }

    template<typename PRIMITIVE>
    void
    extend_valid(size_t size) noexcept {
      return content_;
    }

    std::string
    form() const noexcept {
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
