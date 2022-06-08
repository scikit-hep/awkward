// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/RecordArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/RecordArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  template <typename T, typename I>
  RecordArrayBuilder<T, I>::RecordArrayBuilder(const std::vector<FormBuilderPtr<T, I>>& contents,
                                               const util::RecordLookupPtr recordlookup,
                                               const util::Parameters& parameters,
                                               const std::string& form_key,
                                               const std::string attribute,
                                               const std::string partition)
    : form_recordlookup_(recordlookup),
      parameters_(parameters),
      form_key_(form_key),
      field_index_(0),
      contents_size_((int64_t) contents.size()) {
    for (auto const& content : contents) {
      contents_.push_back(content);
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
      vm_func_.append(contents_.back().get()->vm_func());
      vm_error_.append(contents_.back().get()->vm_error());
    }
    for (auto const& it : *recordlookup) {
      fields_.push_back(it);
    }

    vm_func_name_ = std::string(form_key).append(attribute);

    vm_func_.append(": ")
      .append(vm_func_name_);

    for (auto const& content : contents_) {
      vm_func_.append(" ").append(content.get()->vm_func_name())
        .append(" pause");
    }
    // Remove the last pause
    vm_func_.erase(vm_func_.end() - 6, vm_func_.end());
    vm_func_.append(" ; ");
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::classname() const {
    return "RecordArrayBuilder";
  }

  template <typename T, typename I>
  ssize_t
  RecordArrayBuilder<T, I>::len(const ForthOutputBufferMap& outputs) const {
    if (!contents_.empty()) {
      return contents_.front().get()->len(outputs);
    }
    return 0;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::to_buffers(
    BuffersContainer& container,
    const ForthOutputBufferMap& outputs) const {
    std::stringstream out;
    out << "{\"class\": \"RecordArray\", \"contents\": {";
    for (size_t i = 0;  i < contents().size();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << "" + util::quote(fields_[i]) + ": ";
      out << contents()[i].get()->to_buffers(container, outputs);
    }
    out << "}, ";
    out << this->parameters_as_string(parameters_);
    out << "\"form_key\": \"" + form_key() + "\"}";

    return out.str();
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_output() const {
    return vm_output_;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_output_data() const {
    return vm_output_data_;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_func() const {
    return vm_func_;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_func_name() const {
    return vm_func_name_;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_func_type() const {
    return vm_func_type_;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  template <typename T, typename I>
  const std::string
  RecordArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  int64_t
  RecordArrayBuilder<T, I>::field_index() {
    if (!list_field_index_.empty())
    {
      return field_index_;
    }
    auto index = field_index_;
    field_index_ = (++field_index_ < contents_size_)
                       ? field_index_
                       : field_index_ % contents_size_;
    return index;
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)field_index()].get()->boolean(x, builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)field_index()].get()->int64(x, builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)field_index()].get()->float64(x, builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)field_index()].get()->complex(x, builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)field_index()].get()->bytestring(x, builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)field_index()].get()->string(x, builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
    list_field_index_.emplace_back(field_index_);
    contents_[(size_t)field_index_].get()->begin_list(builder);
  }

  template <typename T, typename I>
  void
  RecordArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
    field_index_ = list_field_index_.back();
    contents_[(size_t)field_index_].get()->end_list(builder);
    list_field_index_.pop_back();
    field_index();
  }

  template <typename T, typename I>
  bool
  RecordArrayBuilder<T, I>::active() {
    if (!list_field_index_.empty()) {
      return contents_[(size_t)list_field_index_.back()].get()->active();
    }
    else {
      for(auto content : contents_) {
        if (content.get()->active()) {
          return true;
        }
      }
    }
    return false;
  }

  template class EXPORT_TEMPLATE_INST RecordArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST RecordArrayBuilder<int64_t, int32_t>;

}
