// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/RecordArrayBuilder.cpp", line)

#include "awkward/typedbuilder/RecordArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/RecordArray.h"

namespace awkward {

  ///
  RecordArrayBuilder::RecordArrayBuilder(const RecordFormPtr& form,
                                         const std::string attribute,
                                         const std::string partition)
    : form_(form),
      field_index_(0),
      contents_size_((int64_t)form.get()->contents().size()),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition) {
    for (auto const& content : form.get()->contents()) {
      contents_.push_back(TypedArrayBuilder::formBuilderFromA(content));
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
      vm_func_.append(contents_.back().get()->vm_func());
      vm_error_.append(contents_.back().get()->vm_error());
    }

    vm_func_name_ = std::string(*form_key_).append(attribute_);

    vm_func_.append(": ")
      .append(vm_func_name_);

    for (auto const& content : contents_) {
      vm_func_.append("\n    ").append(content.get()->vm_func_name())
        .append(" pause");
    }
    // Remove the last pause
    vm_func_.erase(vm_func_.end() - 6, vm_func_.end());
    vm_func_.append("\n;\n\n");
  }

  const std::string
  RecordArrayBuilder::classname() const {
    return "RecordArrayBuilder";
  }

  const ContentPtr
  RecordArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    ContentPtrVec contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      contents.push_back(contents_[i].get()->snapshot(outputs));
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         form_.get()->parameters(),
                                         contents,
                                         form_.get()->recordlookup());
  }

  const FormPtr
  RecordArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  RecordArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  RecordArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  RecordArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  RecordArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  RecordArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  RecordArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  RecordArrayBuilder::vm_error() const {
    return vm_error_;
  }

  int64_t
  RecordArrayBuilder::field_index() {
    return (field_index_ < contents_size_ - 1) ?
      field_index_++ : (field_index_ = 0);
  }

  void
  RecordArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    contents_[(size_t)field_index()].get()->boolean(x, builder);
  }

  void
  RecordArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    contents_[(size_t)field_index()].get()->int64(x, builder);
  }

  void
  RecordArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    contents_[(size_t)field_index()].get()->float64(x, builder);
  }

  void
  RecordArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    contents_[(size_t)field_index()].get()->complex(x, builder);
  }

  void
  RecordArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    contents_[(size_t)field_index()].get()->bytestring(x, builder);
  }

  void
  RecordArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    contents_[(size_t)field_index()].get()->string(x, builder);
  }

  void
  RecordArrayBuilder::begin_list(TypedArrayBuilder* builder) {
    list_field_index_.emplace_back(field_index_);
    contents_[(size_t)field_index_].get()->begin_list(builder);
  }

  void
  RecordArrayBuilder::end_list(TypedArrayBuilder* builder) {
    field_index_ = list_field_index_.back();
    contents_[(size_t)field_index_].get()->end_list(builder);
    list_field_index_.pop_back();
    field_index();
  }

  bool
  RecordArrayBuilder::active() {
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

}
