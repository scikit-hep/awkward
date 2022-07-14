// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LAYOUTBUILDER_H_
#define AWKWARD_LAYOUTBUILDER_H_

#include "GrowableBuffer.h"
#include "utils.h"

#include <tuple>
#include <map>
#include <iostream>

namespace awkward {

  namespace LayoutBuilder {

  template<const char *str>
  class field_name {
  public:
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
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

      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }

      if (std::is_arithmetic<PRIMITIVE>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \""
                  + type_to_name<PRIMITIVE>() +"\"" + params
                  + ", \"form_key\": \"" + form_key.str() + "\" }";
      }
      else if (is_specialization<PRIMITIVE, std::complex>::value) {
        return "{ \"class\": \"NumpyArray\", \"primitive\": \""
                  + type_to_name<PRIMITIVE>() + "\"" + params
                  + ", \"form_key\": \"" + form_key.str() + "\" }";
      }
      else {
        throw std::runtime_error
          ("type " + std::string(typeid(PRIMITIVE).name()) + "is not supported");
      }
    }

  private:
    awkward::GrowableBuffer<PRIMITIVE> data_;
    size_t id_;
    std::string parameters_;
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
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
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
                + content_.form() + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> offsets_;
    BUILDER content_;
    size_t id_;
    std::string parameters_;
  };

  template<bool IS_TUPLE>
  class EmptyRecord {
  public:
    EmptyRecord()
    : length_(0) {
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
    }

    void
    set_id(size_t &id) {
      // id_ = id;
      // id++;
    }

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const noexcept {
      return true;
    }

    void
    append() noexcept {
      length_++;
    }

    void
    extend(size_t size) noexcept {
      length_ += size;
    }

    std::string
    form() const noexcept {
      std::stringstream form_key;
      form_key << "node" << id_;
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }

      if (is_tuple_) {
        return "{ \"class\": \"RecordArray\", \"contents\": []" + params
                  + ", \"form_key\": \"" + form_key.str() + "\" }";
      }
      else {
        return "{ \"class\": \"RecordArray\", \"contents\": {}" + params
                  + ", \"form_key\": \"" + form_key.str() + "\" }";
      }
    }

  private:
    size_t id_;
    std::string parameters_;
    size_t length_;
    bool is_tuple_ = IS_TUPLE;
  };

  template <typename... BUILDERS>
  class Record {
  public:
    Record()
        : contents({new BUILDERS}...) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return (std::get<0>(contents)->builder.length());;
    }

    void
    clear() noexcept {
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
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
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string("\"parameters\": " + parameters_ + ", ");
      }
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
      out << params << "\"form_key\": \"" << form_key.str() << "\" }";
      return out.str();
    }

    std::tuple<BUILDERS*...> contents;

    private:
    size_t id_;
    std::string parameters_;
  };


template <unsigned INITIAL, typename BUILDER>
  class List {
  public:
    List()
        : starts_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , stops_(awkward::GrowableBuffer<int64_t>(INITIAL)) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return starts_.length();
    }

    void
    clear() noexcept {
      starts_.clear();
      stops_.clear();
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
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
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"ListArray\", \"starts\": \"i64\", \"stops\": \"i64\", \"content\": "
                + content_.form() + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> starts_;
    GrowableBuffer<int64_t> stops_;
    BUILDER content_;
    size_t id_;
    std::string parameters_;
  };

template <unsigned INITIAL, typename BUILDER>
  class Indexed {
  public:
    Indexed()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , last_valid_(-1) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return index_.length();
    }

    void
    clear() noexcept {
      last_valid_ = -1;
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const noexcept {
      if (content_.length() != index_.length()) {
        std::cout << "Indexed node" << id_ << " has content length " << content_.length()
                  << " but index length " << index_.length();
        return false;
      }
      else if (content_.length() != last_valid_ + 1) {
        std::cout << "Indexed node" << id_ << " has content length " << content_.length()
                  << " but last valid index is " << last_valid_;
        return false;
      }
      else {
        return content_.is_valid();
      }
    }

    BUILDER*
    append_index() noexcept {
      last_valid_ = content_.length();
      index_.append(last_valid_);
      return &content_;
    }

    BUILDER*
    extend_index(size_t size) noexcept {
      size_t start = content_.length();
      size_t stop = start + size;
      last_valid_ = stop - 1;
      for (size_t i = start; i < stop; i++) {
        index_.append(i);
      }
      return &content_;
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
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"IndexedArray\", \"index\": \"i64\", \"content\": "
                + content_.form() + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> index_;
    BUILDER content_;
    size_t id_;
    std::string parameters_;
    size_t last_valid_;
  };

template <unsigned INITIAL, typename BUILDER>
  class IndexedOption {
  public:
    IndexedOption()
        : index_(awkward::GrowableBuffer<int64_t>(INITIAL))
        , last_valid_(-1) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return index_.length();
    }

    void
    clear() noexcept {
      last_valid_ = -1;
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const noexcept {
      if (content_.length() != last_valid_ + 1) {
        std::cout << "IndexedOption node" << id_ << " has content length "<< content_.length()
                  << " but last valid index is " << last_valid_;
        return false;
      }
      else {
        return content_.is_valid();
      }
    }

    BUILDER*
    append_index() noexcept {
      last_valid_ = content_.length();
      index_.append(last_valid_);
      return &content_;
    }

    BUILDER*
    extend_index(size_t size) noexcept {
      size_t start = content_.length();
      size_t stop = start + size;
      last_valid_ = stop - 1;
      for (size_t i = start; i < stop; i++) {
        index_.append(i);
      }
      return &content_;
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
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"IndexedOptionArray\", \"index\": \"i64\", \"content\": "
                + content_.form() + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int64_t> index_;
    BUILDER content_;
    size_t id_;
    std::string parameters_;
    size_t last_valid_;
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const noexcept {
      return true;
    }

    std::string
    form() const noexcept {
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"EmptyArray\"" + params + " }";
    }

  private:
    size_t id_;
    std::string parameters_;
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const {
      return content_.is_valid();
    }

    BUILDER*
    append_valid() noexcept {
      return &content_;
    }

    BUILDER*
    extend_valid(size_t size) noexcept {
      return &content_;
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
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"UnmaskedArray\", \"content\": " + content_.form()
                + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    BUILDER content_;
    size_t id_;
    std::string parameters_;
  };

  template <bool VALID_WHEN, unsigned INITIAL, typename BUILDER>
  class ByteMasked {
  public:
    ByteMasked()
        : mask_(awkward::GrowableBuffer<int8_t>(INITIAL)) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return mask_.length();
    }

    void
    clear() noexcept {
      mask_.clear();
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const {
      if (content_.length() != mask_.length()) {
        std::cout << "ByteMasked node" << id_ << "has content length " << content_.length()
                  << "but mask length " << mask_.length();
        return false;
      }
      else {
        return content_.is_valid();
      }
    }

    bool
    valid_when() {
      return valid_when_;
    }

    BUILDER*
    append_valid() noexcept {
      mask_.append(valid_when_);
      return &content_;
    }

    BUILDER*
    extend_valid(size_t size) noexcept {
      for (size_t i = 0; i < size; i++) {
        mask_.append(valid_when_);
      }
      return &content_;
    }

    BUILDER*
    append_null() noexcept {
      mask_.append(!valid_when_);
      return &content_;
    }

    BUILDER*
    extend_null(size_t size) noexcept {
      for (size_t i = 0; i < size; i++) {
        mask_.append(!valid_when_);
      }
      return &content_;
    }

    void
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-mask"] = mask_.nbytes();
      content_.buffer_nbytes(names_nbytes);
    }

    void
    to_buffers(int8_t* ptr) const noexcept {
      mask_.concatenate(ptr);
    }

    std::string
    form() const noexcept {
      std::stringstream form_key, form_valid_when;
      form_key << "node" << id_;
      form_valid_when << std::boolalpha << valid_when_;
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"ByteMaskedArray\", \"mask\": \"i8\", \"content\": " + content_.form()
                + ", \"valid_when\": " + form_valid_when.str()
                + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    GrowableBuffer<int8_t> mask_;
    BUILDER content_;
    size_t id_;
    std::string parameters_;
    bool valid_when_ = VALID_WHEN;
  };

  template <bool VALID_WHEN, bool LSB_ORDER, unsigned INITIAL, typename BUILDER>
  class BitMasked {
  public:
    BitMasked()
        : mask_(awkward::GrowableBuffer<uint8_t>(INITIAL))
        , current_byte_(uint8_t(0))
        , current_byte_ref_(mask_.append_and_get_ref(current_byte_))
        , current_index_(0) {
      size_t id = 0;
      set_id(id);
      if (lsb_order_) {
        for(size_t i = 0; i < 8; i++) {
          cast_[i] = 1 << i;
        }
      }
      else {
        for(size_t i = 0; i < 8; i++) {
          cast_[i] = 128 >> i;
        }
      }
    }

    size_t
    length() const noexcept {
      return mask_.length();
    }

    void
    clear() noexcept {
      mask_.clear();
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const {
      if (content_.length() != mask_.length()) {
        std::cout << "BitMasked node" << id_ << "has content length " << content_.length()
                  << "but bit mask length " << mask_.length();
        return false;
      }
      else {
        return content_.is_valid();
      }
    }

    bool
    valid_when() {
      return valid_when_;
    }

    bool
    lsb_order() {
      return lsb_order_;
    }

    BUILDER*
    append_valid() noexcept {
      _append_begin();
      current_byte_ |= cast_[current_index_];
      _append_end();
      return &content_;
    }

    BUILDER*
    extend_valid(size_t size) noexcept {
      for (size_t i = 0; i < size; i++) {
        append_valid();
      }
      return &content_;
    }

    BUILDER*
    append_null() noexcept {
      _append_begin();
      _append_end();
      return &content_;
    }

    BUILDER*
    extend_null(size_t size) noexcept {
      for (size_t i = 0; i < size; i++) {
      append_null();
      }
      return &content_;
    }

    void
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      names_nbytes["node" + std::string(id_) + "-mask"] = mask_.nbytes();
      content_.buffer_nbytes(names_nbytes);
    }

    void
    to_buffers(uint8_t* ptr) const noexcept {
      mask_.concatenate(ptr);
    }

    std::string
    form() const noexcept {
      std::stringstream form_key, form_valid_when, form_lsb_order;
      form_key << "node" << id_;
      form_valid_when << std::boolalpha << valid_when_;
      form_lsb_order << std::boolalpha << lsb_order_;
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"BitMaskedArray\", \"mask\": \"u8\", \"content\": " + content_.form()
                + ", \"valid_when\": " + form_valid_when.str() + ", \"lsb_order\": "
                + form_lsb_order.str() + params + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    void
    _append_begin() {
      if (current_index_ == 8) {
        current_byte_ = uint8_t(0);
        current_byte_ref_ = mask_.append_and_get_ref(current_byte_);
        current_index_ = 0;
      }
    }

    void
    _append_end() {
      current_index_ += 1;
      if (valid_when_) {
        current_byte_ref_ = current_byte_;
      }
      else {
        current_byte_ref_ = ~current_byte_;
      }
    }

    GrowableBuffer<uint8_t> mask_;
    BUILDER content_;
    size_t id_;
    std::string parameters_;
    bool valid_when_ = VALID_WHEN;
    bool lsb_order_ = LSB_ORDER;
    uint8_t current_byte_;
    uint8_t& current_byte_ref_;
    size_t current_index_;
    uint8_t* cast_;
  };

  template <unsigned SIZE, typename BUILDER>
  class Regular {
  public:
    Regular()
        : length_(0) {
      size_t id = 0;
      set_id(id);
    }

    size_t
    length() const noexcept {
      return length_;
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

    std::string parameters() const noexcept {
      return parameters_;
    }

    void
    set_parameters(std::string parameter) noexcept {
      parameters_ = parameter;
    }

    bool is_valid() const noexcept {
      if (content_.length() != length_ * size_) {
        std::cout << "Regular node" << id_ << "has content length " << content_.length()
                  << ", but length " << length_ << " and size " << size_;
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
      length_++;
    }

    void
    buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
      content_.buffer_nbytes(names_nbytes);
    }

    // void
    // to_buffers() const noexcept {

    // }

    std::string
    form() const noexcept {
      std::stringstream form_key;
      form_key << "node" << id_;
      std::string params("");
      if (parameters_ == "") { }
      else {
        params = std::string(", \"parameters\": " + parameters_);
      }
      return "{ \"class\": \"RegularArray\", \"content\": " + content_.form()
                + ", \"size\": " + std::to_string(size_)  + params
                + ", \"form_key\": \"" + form_key.str() + "\" }";
    }

  private:
    BUILDER content_;
    size_t id_;
    size_t length_;
    std::string parameters_;
    size_t size_ = SIZE;
  };

// template <unsigned INITIAL,  typename... BUILDERS>
//   class Union {
//   public:
//     Union()
//         : tags_(awkward::GrowableBuffer<int8_t>(INITIAL))
//         : index_(awkward::GrowableBuffer<int64_t>(INITIAL))
//         , contents_({new BUILDERS}...)
//         , last_valid_index_(-1) {
//       size_t id = 0;
//       set_id(id);
//     }

//     size_t
//     length() const noexcept {
//       return index_.length();
//     }

//     void
//     clear() noexcept {
//       last_valid_ = -1;
//       index_.clear();
//       content_.clear();
//     }

//     BUILDER*
//     content() {
//       return &content_;
//     }

//     void
//     set_id(size_t &id) {
//       id_ = id;
//       id++;
//       content_.set_id(id);
//     }

//     std::string parameters() const noexcept {
//       return parameters_;
//     }

//     bool is_valid() const noexcept {
//       if (content_.length() != index_.length()) {
//         std::cout << "Union node" << id_ << " has content length " << content_.length()
//                   << " but index length " << index_.length();
//         return false;
//       }
//       else if (content_.length() != last_valid_index_ + 1) {
//         std::cout << "Union node" << id_ << " has content length " << content_.length()
//                   << " but last valid index is " << last_valid_index_;
//         return false;
//       }
//       else {
//         return content_.is_valid();
//       }
//     }

//     BUILDER*
//     append_index(int8_t tag) noexcept {
//       auto get_content = [&tag] (auto which_content) {
//       size_t next_index = which_content.length();
//       last_valid_index_[tag] = next_index;
//       tags_.append(tag);
//       index_.append(next_index);};
//       visit_at(contents_, tag, get_content);

//       return &which_content;
//     }

//     void
//     buffer_nbytes(std::map<std::string, size_t> &names_nbytes) const noexcept {
//       names_nbytes["node" + std::string(id_) + "-tags"] = tags_.nbytes();
//       names_nbytes["node" + std::string(id_) + "-index"] = index_.nbytes();
//       content_.buffer_nbytes(names_nbytes);
//     }

//     void
//     to_buffers(int8_t* tags, int64_t* index) const noexcept {
//       tags_.concatenate(tags);
//       index_.concatenate(index);
//     }

//     std::string
//     form() const noexcept {
//       std::stringstream form_key;
//       form_key << "node" << id_;
//       std::string params("");
//       if (parameters_ == "") { }
//       else {
//         params = std::string(", \"parameters\": " + parameters_);
//       }
//       return "{ \"class\": \"UnionArray\", \"tags\": \"i8\", \"index\": \"i64\", \"content\": "
//                 + contents_.form() + params + ", \"form_key\": \"" + form_key.str() + "\" }";
//     }

//   private:
//     GrowableBuffer<int8_t> tags_;
//     GrowableBuffer<int64_t> index_;
//     std::tuple<BUILDERS*...> contents_;
//     size_t id_;
//     std::string parameters_;
//     size_t last_valid_index_;
//   };

  }  // namespace LayoutBuilder
}  // namespace awkward

#endif  // AWKWARD_LAYOUTBUILDER_H_
