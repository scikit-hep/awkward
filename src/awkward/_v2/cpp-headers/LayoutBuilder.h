// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

#include "GrowableBuffer.h"
#include "utils.h"

#include <stdexcept>
#include <tuple>
#include <map>
#include <iostream>

namespace awkward {

  namespace LayoutBuilder {

  template<const char *str>
  struct field_name {
    const char* value = str;
  };

  template<class field_name, class BUILDER>
  class Field {
  public:
    const char* field() {
        return field_.value;
    }

    std::string
    form() const noexcept {
      return builder.form();
    }

  BUILDER builder;

  private:
    field_name field_;
  };

  template <unsigned INITIAL, typename PRIMITIVE>
  class Numpy {
  public:
    Numpy()
        : data_(awkward::GrowableBuffer<PRIMITIVE>(INITIAL)) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return data_.length();
    }

    void
    clear() noexcept {
      data_.clear();
    }

    void
    set_id(size_t &id) {
      id_ = id;
      id++;
    }

    bool is_valid() const noexcept {
      return true;
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
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-data"] = data_.nbytes();
    }

    // void
    // to_buffers(std::map<std::string, PRIMITIVE*> &buffers) const noexcept {
    //   data_.concatenate(buffers["node" + std::string(id_) + "-data"]);
    // }

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
    awkward::GrowableBuffer<PRIMITIVE> data_;
    size_t id_;
  };

  template <unsigned INITIAL, typename BUILDER>
  class ListOffset {
  public:
    ListOffset()
        : offsets_(awkward::GrowableBuffer<int64_t>(INITIAL)) {
      offsets_.append(0);
      size_t id = 0;
      set_id(id);
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

    void
    set_id(size_t &id) {
      id_ = id;
      id++;
      content_.set_id(id);
    }

    bool is_valid() const noexcept {
      if (content_.length() != offsets_.last()) {
        std::cout << "ListOffset node" << id_ << "has content length " << content_.length()
                  << "but last offset " << offsets_.last();
        return false;
      }
      else {
        return content_.is_valid();
      }
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
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-offsets"] = offsets_.nbytes();
      content_.buffer_nbytes(names_nbytes);
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
    size_t id_;
  };

  template <typename... RECORD>
  class Record {
  public:
    Record()
        : contents({new RECORD}...)
        , length_(0) {
      size_t id = 0;
      set_id(id);
    }

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
    set_id(size_t &id) {
      id_ = id;
      id++;
      auto contents_id = [&id](auto record) { record->builder.set_id(id); };
      for (size_t i = 0; i < std::tuple_size<decltype(contents)>::value; i++)
        visit_at(contents, i, contents_id);
    }

    void
    end_record() {
      length_++;
    }

    void
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      auto contents_nbytes = [&names_nbytes](auto record) { record->builder.buffer_nbytes(names_nbytes); };
      for (size_t i = 0; i < std::tuple_size<decltype(contents)>::value; i++)
        visit_at(contents, i, contents_nbytes);
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
    size_t id_;
  };


template <unsigned INITIAL, typename BUILDER>
  class List {
  public:
    List()
        : starts_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , stops_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , length_(0) {
      size_t id = 0;
      set_id(id);
    }

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

    void
    set_id(size_t &id) {
      id_ = id;
      id++;
      content_.set_id(id);
    }

    bool is_valid() const noexcept {
      if (starts_.length() != stops_.length()) {
        std::cout << "List node" << id_ << " has starts length " << starts_.length()
                  << " but stops length " << stops_.length();
        return false;
      }
      else if (stops_.length() > 0 && content_.length() != stops_.last()) {
        std::cout << "List node" << id_ << " has content length " << content_.length()
                  << " but last stops " << stops_.last();
        return false;
      }
      else {
        return content_.is_valid();
      }
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
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-starts"] = starts_.nbytes();
      names_nbytes["node" + std::string(id_) + "-stops"] = stops_.nbytes();
      content_.buffer_nbytes(names_nbytes);
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
    size_t id_;
  };

template <unsigned INITIAL, typename BUILDER>
  class Indexed {
  public:
    Indexed()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL)) {
      size_t id = 0;
      set_id(id);
    }

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
    set_id(size_t &id) {
      id_ = id;
      id++;
      content_.set_id(id);
    }

    bool is_valid() const noexcept {
      if (content_.length() != index_.length()) {
        std::cout << "Indexed node" << id_ << " has content length " << content_.length()
                  << " but index length " << index_.length();
        return false;
      }
      else if (content_.length() != index_.last() + 1) {
        std::cout << "Indexed node" << id_ << " has content length " << content_.length()
                  << " but last valid index is " << index_.last();
        return false;
      }
      else {
        return content_.is_valid();
      }
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
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-index"] = index_.nbytes();
      content_.buffer_nbytes(names_nbytes);
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
    size_t id_;
  };

template <unsigned INITIAL, typename BUILDER>
  class IndexedOption {
  public:
    IndexedOption()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL)) {
      size_t id = 0;
      set_id(id);
    }

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
    set_id(size_t &id) {
      id_ = id;
      id++;
      content_.set_id(id);
    }

    bool is_valid() const noexcept {
      if (content_.length() != index_.last() + 1) {
        std::cout << "IndexedOption node" << id_ << " has content length "<< content_.length()
                  << " but last valid index is " << index_.last();
        return false;
      }
      else {
        return content_.is_valid();
      }
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
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-index"] = index_.nbytes();
      content_.buffer_nbytes(names_nbytes);
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
    size_t id_;
  };

  class Empty {
  public:
    Empty() {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return 0;
    }

    void
    set_id(size_t &id) {
      // id_ = id;
      // id++;
    }

    bool is_valid() const noexcept {
      return true;
    }

    std::string
    form() const noexcept {
      return "{ \"class\": \"EmptyArray\" }";
    }

  private:
    size_t id_;
  };

  template <typename BUILDER>
  class Unmasked {
  public:
    Unmasked() {
      size_t id = 0;
      set_id(id);
    }

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

    void
    set_id(size_t &id) {
      id_ = id;
      id++;
      content_.set_id(id);
    }

    bool is_valid() const {
      return content_.is_valid();
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

    void
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      content_.buffer_nbytes(names_nbytes);
    }

    void
    to_buffers(int64_t* ptr) const noexcept {
      content_.to_buffers(ptr);
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
    size_t id_;
  };

  }  // namespace LayoutBuilder
}  // namespace awkward

#endif  // AWKWARD_LAYOUTBUILDER_H_
